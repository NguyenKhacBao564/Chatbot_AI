
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
from vncorenlp import VnCoreNLP
from dateutil.parser import parse
from datetime import datetime, timedelta
import logging
import re
import os
import calendar
import torch
from extract_location import extract_location
from extract_time import extract_time
from extract_price import extract_price_vn
from pydantic import BaseModel
from google_genAI import get_genai_response
from retrieval import RetrievalPipeline


# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))


class ResetRequest(BaseModel):
    user_id: str = "default_user"

class SessionManager:
    """Quản lý session người dùng với TTL."""
    def __init__(self, ttl_hours=24):
        self.sessions = {}
        self.ttl = timedelta(hours=ttl_hours)

    def get_session(self, user_id):
        if user_id not in self.sessions or self._is_expired(user_id):
            self.sessions[user_id] = {
                "location": None, "time": None, "price": None,
                "last_updated": datetime.now(), "search_history": []
            }
        return self.sessions[user_id]

    def _is_expired(self, user_id):
        session = self.sessions.get(user_id, {})
        last_updated = session.get("last_updated")
        return last_updated and (datetime.now() - last_updated) > self.ttl

    def reset_session(self, user_id):
        self.sessions[user_id] = {
            "location": None, "time": None, "price": None,
            "last_updated": datetime.now(), "search_history": []
        }
        logger.debug(f"Reset session for user_id: {user_id}")

class TourRetrievalPipeline:
    """Pipeline để tìm kiếm và trả lời thông tin tour du lịch."""
    INTENT_LABELS = {
        0: "find_tour_with_location",
        1: "find_tour_with_time",
        2: "find_tour_with_price",
        3: "find_tour_with_location_and_time",
        4: "find_tour_with_location_and_price",
        5: "find_tour_with_time_and_price",
        6: "find_with_all",
        7: "out_of_scope"
     }

    def __init__(self, index_file="faq_index.faiss", metadata_file="faq_metadata.json"):
        # Load Faiss index và metadata
        self.index = faiss.read_index(index_file)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Load SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.retrievalPipeline = RetrievalPipeline()

        # Load PhoBERT cho phân loại ý định
        model_intent_path = os.path.join(current_dir, "phobert_intent_finetuned")
        self.intent_tokenizer = AutoTokenizer.from_pretrained(model_intent_path)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(model_intent_path)
        # self.intent_labels = {0: "find_tour_with_location", 1: "find_tour_with_location_and_time", 2: "find_tour_with_location_and_price", 3: "out_of_scope"}


        jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
        try:
            self.vncorenlp = VnCoreNLP(jar_path, annotators="wseg,pos,ner", max_heap_size='-Xmx2g')
        except Exception as e:
            logger.error(f"Không thể khởi tạo VnCoreNLP: {e}")
            raise

        self.session_manager = SessionManager()
        logger.info("Khởi tạo TourRetrievalPipeline thành công!")

    def extract_intent(self, query):
        try:
            inputs = self.intent_tokenizer(query, return_tensors="pt", max_length=128, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            return self.INTENT_LABELS[predicted_class]
        except Exception as e:
            logger.error(f"Lỗi phân loại ý định: {e}")
            return "out_of_scope"

    def extract_entities(self, query,intent, user_id="default_user"):
        session = self.session_manager.get_session(user_id)
        location = session["location"]
        time = session["time"]
        price = session["price"]


        if intent in ["find_tour_with_location", "find_tour_with_location_and_time", "find_tour_with_location_and_price", "find_with_all"]:
            extracted_location = extract_location(query)
            if extracted_location != "None":
                location = extracted_location
                session["location"] = location
                logger.debug(f"Extracted location: {location}")

        if intent in  ["find_tour_with_location_and_time", "find_tour_with_time", "find_tour_with_time_and_price", "find_with_all"]:
            extracted_time = extract_time(query)
            if extracted_time != "None":
                time = extracted_time
                session["time"] = time
                logger.debug(f"Extracted time: {time}")

        if intent in ["find_tour_with_location_and_price", "find_tour_with_price", "find_tour_with_time_and_price", "find_with_all"]:
            extracted_price = extract_price_vn(query)
            if extracted_price != "None":
                price = extracted_price
                session["price"] = price
                logger.debug(f"Extracted price: {price}")
       
        session["last_updated"] = datetime.now()
        session["search_history"].append({"query": query})
        logger.debug(f"Extracted entities: location={location}, time={time}, price={price}")
        return {"location": location, "time": time, "price": price}

    def reset_session(self, user_id):
            self.session_manager.reset_session(user_id)
            logger.debug(f"Session reset for user_id: {user_id}")

    def get_faq_response(self, query, k=1):
        try:
            respond = self.retrievalPipeline.get_retrieved_context(query, top_k=k)
            response_text = get_genai_response(
                "Kiểm tra xem câu query này: '" + query + "' và câu trả lời này: '" + respond + 
                "' có phù hợp không, nếu phù hợp thì trả lời y như câu trả lời đó, nếu không thì chỉ cần trả lời là không thể trả lời câu hỏi này, và hỏi xem khách hàng có muốn hỏi về tour không? "
            )
            return {
                "response": response_text,
            }
        except Exception as e:
            logger.error(f"Lỗi tìm FAQ với KNN: {e}")
            return {
                "response": "Dạ, em chưa hiểu ý bạn. Bạn có thể hỏi về tour hoặc đặt phòng khách sạn không ạ?",
            }

    def get_tour_response(self, query, user_id="default_user"):
        intent = self.extract_intent(query)
        print("intent: ", intent)
        if intent == "out_of_scope":
            faq_response = self.get_faq_response(query)
            return {
                    "status": "faq",
                    "response": f"{faq_response['response']}",
                    "location": "None",
                    "time": "None",
                    "price": "None"
                }
        
        info = self.extract_entities(query, intent, user_id)

        missing_info = []
        if not info["location"]:
            missing_info.append("Điểm đến (ví dụ: Đà Lạt, Hà Nội, Phú Quốc,...)")
        if not info["time"]:
            missing_info.append("Thời gian khởi hành (ví dụ: tháng 12, 2 tuần tới,...)")
        if not info["price"]:
            missing_info.append("Giá (ví dụ: dưới 5 triệu, khoảng 2 triệu,...)")

        if missing_info:
            prompt = (
                f"Người dùng đã cung cấp thông tin về tour du lịch: "
                f"Địa điểm: {info['location'] or 'chưa có'}, "
                f"Thời gian: {info['time'] or 'chưa có'}, "
                f"Giá: {info['price'] or 'chưa có'}. "
                f"Vui lòng viết một câu trả lời lịch sự bằng tiếng Việt, xác nhận thông tin đã nhận và yêu cầu cung cấp các thông tin còn thiếu (thông tin sẽ cung cấp sau không cần nói ra cụ thể) "
                f"Nói kiểu câu cuối cùng có dấu :"
                f"Ví dụ: 'Dạ, em đã ghi nhận thông tin quý khách muốn tìm tour đến [địa điểm]/[thời gian khởi hành]/[giá tour]. Xin quý khách vui lòng cho em biết thêm thông tin như:'"
            )
            try:
                response_text = get_genai_response(prompt)
                return {
                    "status": "missing_info",
                    "response": f"{response_text}\n- " + "\n- ".join(missing_info),
                    "location": info['location'] or "None",
                    "time": info['time'] or "None",
                    "price": info['price'] or "None"
                }
            except Exception as e:
                logger.error(f"Lỗi gọi Gemini API: {e}")
                return {
                    "status": "missing_info",
                    "response": "Dạ, để tìm tour phù hợp, em cần bạn cung cấp thêm:\n- " + "\n- ".join(missing_info),
                    "location": info['location'] or "None",
                    "time": info['time'] or "None",
                    "price": info['price'] or "None"
                }
        # Đủ thông tin thì tìm kiếm
        prompt = (
                f"Đã có các thông tin về tour du lịch: "
                f"Địa điểm: {info['location']}, "
                f"Thời gian: {info['time']}, "
                f"Giá: {info['price']}. "
                f"Vui lòng viết một câu trả lời lịch sự bằng tiếng Việt, xác nhận thông tin đã nhận và đưa ra câu mở đầu trước khi liệt kê các tour đã tìm được. "
                f"Nói kiểu câu cuối cùng có dấu :"
                f"Ví dụ: 'Dạ, em đã tìm được một số tour phù hợp như: "
            )
        response_text = get_genai_response(prompt)
        response_obj = {
            "status": "success",
            "response": f"{response_text}",
            "location": info['location'] or "None",
            "time": info['time'] or "None",
            "price": info['price'] or "None"
        }
        self.reset_session("default_user")
        return response_obj




# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     pipeline = TourRetrievalPipeline(
#         index_file=os.path.join(current_dir, "faq_index.faiss"),
#         metadata_file=os.path.join(current_dir, "faq_metadata.json")
#     )
#     user_id = "test_user"
#     while True:
#         user_query = input("Bạn: ")
#         if user_query.lower() in ["exit", "quit"]:
#             break
#         response = pipeline.get_tour_response(user_query, user_id=user_id)
#         print(f"Bot: {response}")









# import os
# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
# from subprocess import Popen, PIPE
# from dateutil.parser import parse
# from datetime import datetime, timedelta
# import logging
# import re
# import calendar
# import torch
# import tempfile

# # Cấu hình logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# class TourRetrievalPipeline:
#     def __init__(self, index_file="faq_index.faiss", metadata_file="faq_metadata.json"):
#         # Load Faiss index và metadata
#         self.index = faiss.read_index(index_file)
#         with open(metadata_file, 'r', encoding='utf-8') as f:
#             self.metadata = json.load(f)

#         # Load SentenceTransformer
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#         # Load PhoBERT cho phân loại ý định
#         self.intent_tokenizer = AutoTokenizer.from_pretrained("./phobert_intent_finetuned")
#         self.intent_model = AutoModelForSequenceClassification.from_pretrained("./phobert_intent_finetuned")
#         self.intent_labels = {0: "find_tour", 1: "book_room_with_num_people", 2: "book_room_with_number"}

#         # Load PhoBERT cho NER
#         self.ner_tokenizer = AutoTokenizer.from_pretrained("./phobert_ner_finetuned")
#         self.ner_model = AutoModelForTokenClassification.from_pretrained("./phobert_ner_finetuned")
#         self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, aggregation_strategy="simple")
#         self.label_list = ["O", "B-LOC", "B-DATE", "I-DATE", "B-num_people", "B-num_rooms"]
#         self.id2label = {i: label for i, label in enumerate(self.label_list)}

#        # Khởi tạo VnCoreNLP như tiến trình con
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         self.jar_path = os.path.join(current_dir, "VnCoreNLP-1.2.jar")
#         models_path = os.path.join(current_dir, "models", "wordsegmenter", "wordsegmenter.rdr")
#         if not os.path.exists(self.jar_path):
#             raise FileNotFoundError(f"File {self.jar_path} không tồn tại!")
#         if not os.path.exists(models_path):
#             raise FileNotFoundError(f"File mô hình {models_path} không tồn tại! Vui lòng đảm bảo thư mục 'models' nằm cùng cấp với file .jar.")
#         logger.info(f"Đang chuẩn bị sử dụng VnCoreNLP với file: {self.jar_path}")

#         # Quản lý session
#         self.sessions = {}
#         logger.info("Khởi tạo TourRetrievalPipeline thành công!")

#     def tokenize_with_vncorenlp(self, query):
#         print("bắt đầu tokenize")
#         # Tạo file tạm để lưu query
#         with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8', suffix='.txt', delete=False) as input_file:
#             input_file.write(query)
#             input_file_path = input_file.name
        
#         # Tạo file tạm cho đầu ra
#         output_file_path = input_file_path + '.out'

#         # Chạy VnCoreNLP như tiến trình con
#         command = [
#             "java", "-Xmx2g", "-jar", self.jar_path,
#             "-fin", input_file_path,
#             "-fout", output_file_path,
#             "-annotators", "wseg,pos,ner"
#         ]
#         process = Popen(command, stdout=PIPE, stderr=PIPE)
#         stdout, stderr = process.communicate()
#         if process.returncode != 0:
#             logger.error(f"Lỗi khi chạy VnCoreNLP: {stderr.decode()}")
#             raise Exception(f"Lỗi khi chạy VnCoreNLP: {stderr.decode()}")

#         # Đọc kết quả
#         with open(output_file_path, 'r', encoding='utf-8') as f:
#             result = f.read()
#         logger.debug(f"Kết quả tokenize từ VnCoreNLP: {result}")

#         # Xóa file tạm
#         os.unlink(input_file_path)
#         os.unlink(output_file_path)

#         return result.splitlines()  # Trả về danh sách các dòng token

#     def extract_intent(self, query):
#         inputs = self.intent_tokenizer(query, return_tensors="pt", max_length=128, truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = self.intent_model(**inputs)
#         logits = outputs.logits
#         predicted_class = torch.argmax(logits, dim=1).item()
#         print("predicted_class: ",predicted_class)
#         intent = self.intent_labels[predicted_class]
#         logger.debug(f"Detected intent: {intent}")
#         return intent

#     def extract_entities(self, query, user_id="default_user"):
#         if user_id not in self.sessions:
#             self.sessions[user_id] = {
#                 "location": None, "time": None, "time_display": None,
#                 "departure": None, "num_people": None, "num_rooms": None
#             }

#         session = self.sessions[user_id]

#         # Sửa lỗi chính tả
#         query = query.replace("tron", "trong")
#         logger.debug(f"Query sau khi sửa lỗi chính tả: {query}")

#         # Tokenize bằng VnCoreNLP
#         tokens = self.tokenize_with_vncorenlp(query)
#         tokens = [word for sentence in tokens for word in sentence]  # Giả định mỗi dòng là một câu đã phân đoạn
#         logger.debug(f"Tokens: {tokens}")

#         # Trích xuất thực thể bằng PhoBERT NER
#         entities = self.ner_pipeline(query)
#         logger.debug(f"Entities: {entities}")

#         location = session["location"]
#         time = session["time"]
#         time_display = session["time_display"]
#         departure = session["departure"]
#         num_people = session["num_people"]
#         num_rooms = session["num_rooms"]

#         # Chuyển đổi từ tiếng Việt sang số (nếu cần)
#         word_to_num = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10}

#         # Trích xuất từ entities
#         for entity in entities:
#             entity_label = entity["entity_group"]
#             entity_text = entity["word"]
#             if entity_label == "B-LOC":
#                 if not location:
#                     location = entity_text
#                 else:
#                     departure = entity_text
#             elif entity_label == "B-DATE":
#                 try:
#                     time = parse(entity_text, fuzzy=True)
#                     time_display = entity_text
#                     logger.debug(f"Parsed time as datetime: {time}")
#                 except ValueError as e:
#                     logger.warning(f"Không parse được {entity_text} thành datetime: {e}")
#                     time_display = entity_text
#             elif entity_label == "B-num_people":
#                 num_people = word_to_num.get(entity_text.lower(), int(entity_text))
#             elif entity_label == "B-num_rooms":
#                 num_rooms = word_to_num.get(entity_text.lower(), int(entity_text))

#         # Logic thủ công bổ sung cho thời gian
#         query_lower = query.lower()
#         month_names = {
#             "tháng 1": 1, "tháng 2": 2, "tháng 3": 3, "tháng 4": 4, "tháng 5": 5,
#             "tháng 6": 6, "tháng 7": 7, "tháng 8": 8, "tháng 9": 9, "tháng 10": 10,
#             "tháng 11": 11, "tháng 12": 12
#         }
#         for month_name, month_num in month_names.items():
#             if month_name in query_lower and not time:
#                 current_year = datetime.now().year
#                 if "này" in query_lower and month_num < datetime.now().month:
#                     current_year += 1
#                 start_date = datetime(current_year, month_num, 1)
#                 end_date = datetime(current_year, month_num, calendar.monthrange(current_year, month_num)[1])
#                 time = (start_date, end_date)
#                 time_display = month_name
#                 break

#         date_pattern = r'\d{1,2}/\d{1,2}(?:/\d{2,4})?'
#         match = re.search(date_pattern, query)
#         if match and not time:
#             date_str = match.group(0)
#             try:
#                 if len(date_str.split('/')) == 2:
#                     date_str = f"{date_str}/{datetime.now().year}"
#                 parsed_date = parse(date_str, dayfirst=True)
#                 time = (parsed_date, parsed_date)
#                 time_display = date_str
#                 logger.debug(f"Parsed custom date {date_str} as: {parsed_date}")
#             except ValueError as e:
#                 logger.warning(f"Không parse được {date_str} thành datetime: {e}")
#                 time_display = date_str

#         # Cập nhật session
#         if location:
#             session["location"] = location
#         if time:
#             session["time"] = time
#             session["time_display"] = time_display
#         if departure:
#             session["departure"] = departure
#         if num_people:
#             session["num_people"] = num_people
#         if num_rooms:
#             session["num_rooms"] = num_rooms

#         logger.debug(f"Extracted entities: location={location}, time={time}, time_display={time_display}, departure={departure}, num_people={num_people}, num_rooms={num_rooms}")
#         return {
#             "location": location, "time": time, "time_display": time_display,
#             "departure": departure, "num_people": num_people, "num_rooms": num_rooms
#         }

#     def get_tour_response(self, query, top_k=3, user_id="default_user"):
#         intent = self.extract_intent(query)
#         info = self.extract_entities(query, user_id)
#         session = self.sessions[user_id]

#         if intent == "find_tour":
#             if not info["location"]:
#                 return "Dạ, để tìm tour phù hợp, em cần bạn cung cấp thêm:\n- Điểm đến (ví dụ: Đà Lạt, Hà Nội, Phú Quốc)\nBạn cho em biết điểm đến nha!"
#             if not info["time"]:
#                 return "Dạ, em cần bạn cung cấp thêm:\n- Thời gian khởi hành (ví dụ: tháng 12, 2 tuần tới)\nBạn cho em biết thời gian nha!"
#             if not info["departure"]:
#                 return "Dạ, em cần bạn cung cấp thêm:\n- Địa điểm khởi hành (ví dụ: từ TP.HCM, Hà Nội)\nBạn cho em biết điểm khởi hành nha!"

#             query_description = f"Tour tại {info['location']}"
#             query_embedding = self.model.encode([query_description], show_progress_bar=False)
#             query_embedding = np.array(query_embedding).astype("float32")

#             distances, indices = self.index.search(query_embedding, top_k)
#             logger.debug(f"Faiss search results - distances: {distances}, indices: {indices}")

#             start_date = info["time"][0] if isinstance(info["time"], tuple) else info["time"]
#             end_date = info["time"][1] if isinstance(info["time"], tuple) else info["time"]
#             if isinstance(end_date, str):
#                 try:
#                     end_date = parse(end_date, fuzzy=True) + timedelta(days=14)
#                 except ValueError:
#                     end_date = start_date + timedelta(days=14)

#             filtered_tours = []
#             for idx in indices[0]:
#                 if idx < len(self.metadata):
#                     tour = self.metadata[idx]
#                     if tour["destination"].lower() != info["location"].lower():
#                         continue
#                     tour_dates = [datetime.strptime(d, "%Y-%m-%d") for d in tour["start_dates"]]
#                     if not any(start_date <= d <= end_date for d in tour_dates):
#                         continue
#                     if info["departure"] and info["departure"].lower() not in [d.lower() for d in tour["departure"]]:
#                         continue
#                     filtered_tours.append(tour)

#             if not filtered_tours:
#                 response = f"Xin lỗi, em chưa tìm thấy tour nào đến {info['location']} trong thời gian {info['time_display']}"
#                 if info["departure"]:
#                     response += f" từ {info['departure']}"
#                 response += ". Bạn muốn thử địa điểm khác hoặc thời gian khác không nè?"
#                 return response

#             response = f"Dạ, bạn muốn khám phá {info['location']} trong {info['time_display']}"
#             if info["departure"]:
#                 response += f" từ {info['departure']}"
#             response += "! 😊 Dưới đây là một số tour gợi ý:\n"
#             for tour in filtered_tours:
#                 response += f"\n- **{tour['name']}**\n"
#                 response += f"  - Số ngày: {tour['duration']}\n"
#                 response += f"  - Ngày khởi hành: {', '.join(tour['start_dates'])}\n"
#                 response += f"  - Giá từ: {tour['price']:,}₫\n"
#                 response += f"  - Điểm nổi bật: {', '.join(tour['highlights'])}.\n"
#                 response += f"  - Chi tiết: {tour['details_url']}\n"
#             response += "\nBạn ưng tour nào không? 😊 Muốn em gửi chi tiết lịch trình hay hỗ trợ đặt tour luôn nè?"

#             # Xóa session sau khi hoàn thành
#             self.sessions.pop(user_id, None)
#             return response

#         elif intent == "book_room_with_num_people":
#             if not info["location"]:
#                 return "Dạ, để đặt phòng, em cần bạn cung cấp thêm:\n- Điểm đến (ví dụ: Đà Lạt, Hà Nội, Phú Quốc)\nBạn cho em biết điểm đến nha!"
#             if not info["time"]:
#                 return "Dạ, em cần bạn cung cấp thêm:\n- Thời gian lưu trú (ví dụ: tháng 12, 2 tuần tới)\nBạn cho em biết thời gian nha!"
#             if not info["num_people"]:
#                 return "Dạ, em cần biết số lượng người để đặt phòng. Bạn cho em biết số người nha!"

#             response = f"Dạ, bạn muốn đặt phòng tại {info['location']} cho {info['num_people']} người trong {info['time_display']}."
#             if info["departure"]:
#                 response += f" Điểm khởi hành từ {info['departure']}."
#             response += "\nEm sẽ tìm khách sạn phù hợp và gợi ý tour đi kèm nếu cần. Bạn chờ em một chút nha! 😊"

#             # Xử lý giả lập tìm khách sạn
#             response += "\nGợi ý khách sạn:\n- **Khách sạn Thanh Lịch** (Huế)\n  - Phòng đôi: 500,000₫/đêm\n  - Phòng gia đình: 800,000₫/đêm\n  - Đặt ngay: [Link đặt phòng]\n"
#             response += "\nBạn có muốn em tìm thêm tour đi kèm không nè?"

#             self.sessions.pop(user_id, None)
#             return response

#         elif intent == "book_room_with_number":
#             if not info["location"]:
#                 return "Dạ, để đặt phòng, em cần bạn cung cấp thêm:\n- Điểm đến (ví dụ: Đà Lạt, Hà Nội, Phú Quốc)\nBạn cho em biết điểm đến nha!"
#             if not info["time"]:
#                 return "Dạ, em cần bạn cung cấp thêm:\n- Thời gian lưu trú (ví dụ: tháng 12, 2 tuần tới)\nBạn cho em biết thời gian nha!"
#             if not info["num_rooms"]:
#                 return "Dạ, em cần biết số lượng phòng để đặt. Bạn cho em biết số phòng nha!"

#             response = f"Dạ, bạn muốn đặt {info['num_rooms']} phòng tại {info['location']} trong {info['time_display']}."
#             if info["departure"]:
#                 response += f" Điểm khởi hành từ {info['departure']}."
#             response += "\nEm sẽ tìm khách sạn phù hợp và gợi ý tour đi kèm nếu cần. Bạn chờ em một chút nha! 😊"

#             # Xử lý giả lập tìm khách sạn
#             response += "\nGợi ý khách sạn:\n- **Khách sạn Thanh Lịch** (Huế)\n  - Phòng đôi: 500,000₫/đêm\n  - Phòng gia đình: 800,000₫/đêm\n  - Đặt ngay: [Link đặt phòng]\n"
#             response += "\nBạn có muốn em tìm thêm tour đi kèm không nè?"

#             self.sessions.pop(user_id, None)
#             return response

#         else:
#             return "Dạ, em chưa hiểu ý bạn. Bạn có thể hỏi về tour hoặc đặt phòng khách sạn không ạ?"

# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     pipeline = TourRetrievalPipeline(
#         index_file=os.path.join(current_dir, "faq_index.faiss"),
#         metadata_file=os.path.join(current_dir, "faq_metadata.json")
#     )
#     user_id = "test_user"
#     while True:
#         user_query = input("Bạn: ")
#         if user_query.lower() in ["exit", "quit"]:
#             break
#         response = pipeline.get_tour_response(user_query, user_id=user_id)
#         print(f"Bot: {response}")


