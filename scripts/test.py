# from vncorenlp import VnCoreNLP
# import os
# import logging
# from transformers import AutoTokenizer
# import torch
# import json
# import re
# import numpy as np
import torch
print(torch.__version__)  # Phiên bản PyTorch, ví dụ: 2.5.0+cu124
print(torch.cuda.is_available())  # Phải trả về True
print(torch.cuda.get_device_name(0))  # Phải hiển thị "NVIDIA GeForce RTX 4060"
print(torch.version.cuda)  # Phải hiển thị phiên bản CUDA, ví dụ: 12.4

# # Khởi tạo VnCoreNLP
# try:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
#     vncorenlp = VnCoreNLP(jar_path, annotators="wseg,ner", max_heap_size='-Xmx2g')
# except Exception as e:
#     print(f"Lỗi khi khởi tạo VnCoreNLP: {e}")
#     exit(1)

# def preprocess_text(text):
#     """Phân đoạn từ và nhận diện thực thể bằng VnCoreNLP."""
#     try:
#         # Gọi VnCoreNLP để xử lý văn bản
#         annotated_text = vncorenlp.annotate(text)
#         print(f"Annotated text: {annotated_text}")
#         return annotated_text
#     except Exception as e:
#         print(f"Lỗi khi xử lý văn bản: {e}")
#         return None

# def extract_ner(annotated_text):
#     """Trích xuất thực thể NER từ đầu ra của VnCoreNLP."""
#     entities = []
#     for sentence in annotated_text['sentences']:
#         current_entity = []
#         current_label = None
        
#         for token in sentence:
#             word = token['form']
#             ner_label = token['nerLabel']
            
#             if ner_label != 'O':
#                 if current_label == ner_label:
#                     current_entity.append(word)
#                 else:
#                     if current_entity:
#                         entities.append({
#                             "text": " ".join(current_entity),
#                             "label": current_label
#                         })
#                     current_entity = [word]
#                     current_label = ner_label
#             else:
#                 if current_entity:
#                     entities.append({
#                         "text": " ".join(current_entity),
#                         "label": current_label
#                     })
#                     current_entity = []
#                     current_label = None
        
#         # Kiểm tra nếu còn thực thể cuối câu
#         if current_entity:
#             entities.append({
#                 "text": " ".join(current_entity),
#                 "label": current_label
#             })
    
#     return entities

# def extract_price_time(text):
#     """Trích xuất PRICE và TIME bằng regex."""
#     # Regex cho giá
#     price_pattern = r'(trên|dưới|từ|khoảng)?\s*(\d+\.?\d*)\s*(triệu|nghìn|đồng|VND)'
#     prices = re.findall(price_pattern, text, re.IGNORECASE)
    
#     # Regex cho thời gian
#     time_pattern = r'(tháng\s*\d+|ngày\s*\d+/\d+|năm\s*\d{4}|\b(?:thứ|chủ\s*nhật|cuối\s*tuần)\s*(này|trước|sau)?\b)'
#     times = re.findall(time_pattern, text, re.IGNORECASE)
    
#     entities = []
#     for price in prices:
#         price_text = f"{price[0]} {price[1]} {price[2]}" if price[0] else f"{price[1]} {price[2]}"
#         entities.append({"text": price_text.strip(), "label": "PRICE"})
    
#     for time in times:
#         time_text = time[0]
#         entities.append({"text": time_text, "label": "TIME"})
    
#     return entities

# def combine_entities(ner_entities, price_time_entities):
#     """Kết hợp thực thể NER và PRICE/TIME."""
#     return ner_entities + price_time_entities

# # Xử lý câu ví dụ
# text = "tôi muốn du lịch ở Đà Nẵng vào tháng 12 năm 2023 với giá khoảng 3 triệu đồng"
# # Phân đoạn và nhận diện thực thể
# annotated_text = preprocess_text(text)
# if annotated_text:
#     # Trích xuất NER
#     ner_entities = extract_ner(annotated_text)
#     # Trích xuất PRICE và TIME
#     price_time_entities = extract_price_time(text)
#     # Kết hợp thực thể
#     final_entities = combine_entities(ner_entities, price_time_entities)
    
#     # In kết quả
#     print(json.dumps(final_entities, indent=2, ensure_ascii=False))
# else:
#     print("Không thể xử lý văn bản.")

# import os
# import json
# from vncorenlp import VnCoreNLP

# # Khởi tạo VnCoreNLP
# try:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
#     vncorenlp = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx2g')
# except Exception as e:
#     print(f"Lỗi khi khởi tạo VnCoreNLP: {e}")
#     exit(1)

# def preprocess_text(text):
#     """Phân đoạn từ bằng VnCoreNLP."""
#     try:
#         tokens = vncorenlp.tokenize(text)
#         tokens = [word for sentence in tokens for word in sentence]
#         return tokens
#     except Exception as e:
#         print(f"Lỗi khi xử lý văn bản: {e}")
#         return []

# def create_training_data(sentences, output_file="train_ner.txt"):
#     """Tạo file dữ liệu huấn luyện NER theo định dạng BIO."""
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for sentence, labels in sentences:
#             # Phân đoạn từ
#             tokens = preprocess_text(sentence)
#             if len(tokens) != len(labels):
#                 print(f"Lỗi: Số token ({len(tokens)}) không khớp với số nhãn ({len(labels)}) cho câu: {sentence}")
#                 continue
            
#             # Ghi token và nhãn vào file
#             for token, label in zip(tokens, labels):
#                 f.write(f"{token} {label}\n")
#             f.write("\n")  # Dòng trống giữa các câu


# # Danh sách câu ví dụ và nhãn tương ứng
# training_sentences = [
# ]

# # Tạo file dữ liệu huấn luyện
# create_training_data(training_sentences, output_file="train_ner.txt")

# print("Đã tạo file train_ner.txt thành công!")



# from dateutil.parser import parse
# query = "Hôm nay là ngày 25/12/2023"
# date = parse(query, fuzzy=True)
# print(date)  # Output: 2023-12-25
# import requests

# def extract_time_duckling(text, locale='vi_VN', duckling_url='http://localhost:8000/parse'):
#     payload = {
#         'locale': locale,
#         'text': text
#     }
#     try:
#         response = requests.post(duckling_url, data=payload)
#         response.raise_for_status()
#         data = response.json()
        
#         # Lọc thực thể thời gian
#         time_entities = [ent for ent in data if ent['dim'] == 'time']
#         return time_entities
#     except Exception as e:
#         print(f"Lỗi khi gọi Duckling: {e}")
#         return []

# # Ví dụ
# query = "tôi muốn đặt tour vào ngày 3 tháng 5"
# results = extract_time_duckling(query)

# for r in results:
#     print(f"Phát hiện thời gian: {r['body']} → {r['value']['value']}")



























# import os
# from vncorenlp import VnCoreNLP

# # Khởi tạo VnCoreNLP
# try:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
#     vncorenlp = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx2g')
# except Exception as e:
#     print(f"Lỗi khi khởi tạo VnCoreNLP: {e}")
#     exit(1)

# def preprocess_text(text):
#     """Phân đoạn từ bằng VnCoreNLP."""
#     try:
#         tokens = vncorenlp.tokenize(text)
#         tokens = [word for sentence in tokens for word in sentence]
#         return tokens
#     except Exception as e:
#         print(f"Lỗi khi xử lý văn bản: {e}")
#         return []

# def append_tokenized_data(sentences, output_file="tokenized_data.txt"):
#     """Ghi thêm các token đã phân đoạn vào file hiện có, mỗi token trên một dòng."""
#     # Mở file ở chế độ append ('a')
#     with open(output_file, 'a', encoding='utf-8') as f:
#         # Nếu file không rỗng, thêm dòng trống để đảm bảo phân tách với dữ liệu cũ
#         if os.path.getsize(output_file) > 0:
#             f.write("\n")
        
#         for sentence in sentences:
#             # Phân đoạn từ
#             tokens = preprocess_text(sentence)
#             if not tokens:
#                 print(f"Lỗi: Không thể phân đoạn câu: {sentence}")
#                 continue
            
#             # Ghi token vào file
#             for token in tokens:
#                 f.write(f"{token}\n")
#             f.write("\n")  # Dòng trống giữa các câu

# # Danh sách câu mới để thêm vào
# new_sentences = [
#     "tôi muốn tìm tour đi Phú Quốc giá khoảng 4 triệu",
#     "có tour nào đi Đà Lạt dưới 3 triệu không",
#     "tour Sapa 5 ngày 4 đêm giá bao nhiêu",
#     "mình cần tour Nha Trang giá 2 triệu rưỡi",
#     "tour miền Trung kết hợp Huế và Đà Nẵng giá 5 triệu",
#     "có tour đi Đà Nẵng Hội An vào tháng 7 không",
#     "tôi muốn đi Hạ Long với giá dưới 2 triệu",
#     "tour Côn Đảo hiện tại có giá tốt không",
#     "tour 3 ngày 2 đêm ở Cần Thơ giá 2 triệu",
#     "muốn đặt tour Hà Giang cho 2 người khoảng 6 triệu",
#     "tour đi đảo Nam Du giá rẻ nhất bao nhiêu",
#     "tôi quan tâm tour du lịch Phan Thiết 3 triệu",
#     "tour Ninh Bình Tràng An giá khoảng 1.5 triệu",
#     "có tour Hà Nội - Mộc Châu - Sơn La không",
#     "tour Lý Sơn hiện có giá ưu đãi gì không",
#     "mình muốn tour 5 triệu đi Huế hoặc Đà Lạt",
#     "đặt tour nước ngoài như Thái Lan giá 6 triệu",
#     "tour nghỉ dưỡng ở Phú Yên giá thế nào",
#     "tôi cần tour từ TP.HCM đi Hà Giang 5 triệu",
#     "tour Bà Nà Hills 1 ngày giá bao nhiêu",
#     "đi Quy Nhơn với tour trọn gói dưới 4 triệu",
#     "tour Cà Mau 2 ngày có giá dưới 3 triệu không",
#     "tôi muốn tour Vũng Tàu cuối tuần giá rẻ",
#     "tour Hà Nội - Tam Đảo 2 ngày giá bao nhiêu",
#     "có tour khám phá Tây Bắc với giá 6 triệu không",
#     "tour đi Đắk Lắk hoặc Gia Lai trong 4 ngày",
#     "muốn tìm tour du lịch Hà Nội tháng 12",
#     "tour Phan Rang - Tháp Chàm giá dưới 2 triệu",
#     "tôi quan tâm tour Phú Quốc vào dịp lễ giá tốt",
#     "tour miền Tây 3 ngày giá tầm 3 triệu có không"
# ]


# # Ghi thêm dữ liệu vào file
# append_tokenized_data(new_sentences, output_file="tokenized_data.txt")

# print("Đã ghi thêm dữ liệu vào file tokenized_data.txt thành công!")

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
# vncorenlp = VnCoreNLP(jar_path, annotators="wseg,ner", max_heap_size='-Xmx2g')
# extended_data = []

# # Tạo dữ liệu nhận biết địa điêmr
# locations = [
#     "An Giang", "Bà Rịa - Vũng Tàu", "Bạc Liêu", "Bắc Giang", "Bắc Kạn", "Bắc Ninh",
#     "Bến Tre", "Bình Dương", "Bình Định", "Bình Phước", "Bình Thuận", "Cà Mau",
#     "Cao Bằng", "Cần Thơ", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
#     "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương",
#     "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
#     "Kon Tum", "Lai Châu", "Lạng Sơn", "Lào Cai", "Lâm Đồng", "Long An", "Nam Định",
#     "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam",
#     "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình",
#     "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "TP. Hồ Chí Minh", "Trà Vinh",
#     "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"
# ]
# query = ""
# for loc in locations:
#     segmented_query = vncorenlp.tokenize(loc)
#     # segmented_query = " ".join(word for sentence in segmented_query for word in sentence)
#     for sentence in segmented_query:
#         for word in sentence:
#             extended_data.append({"word": word, "label": "LOC"})

# from dateutil.parser import parse
# from datetime import datetime, timedelta
# from underthesea import word_tokenize, ner
# query="tôi muốn tìm tour ở TP HCM vào tháng 12 năm 2023"
# # Tách từ
# tokens = word_tokenize(query)
# # Nhận diện thực thể
# entities = ner(query)
# print(f"tokens: {tokens}")
# print(f"entities: {entities}")
# import spacy
# from spacy.language import Language

# # Tạo pipeline tiếng Việt trống
# nlp = spacy.blank("vi")

# # Thêm entity_ruler vào pipeline
# ruler = nlp.add_pipe("entity_ruler")

# # Định nghĩa các mẫu cho TIME và MONEY
# patterns = [
#     # MONEY: "3 triệu", "5 nghìn đồng", "10 tỷ"
#     {
#         "label": "MONEY",
#         "pattern": [
#             {"TEXT": {"REGEX": r"\d+"}},  # Số
#             {"TEXT": {"IN": ["triệu", "nghìn", "đồng", "tỷ"]}}  # Đơn vị tiền
#         ]
#     },
#     {
#         "label": "MONEY",
#         "pattern": [
#             {"TEXT": {"REGEX": r"\d+"}},  # Số
#             {"TEXT": {"IN": ["triệu", "nghìn", "đồng", "tỷ"]}},
#             {"TEXT": "đồng", "OP": "?"}  # "đồng" là tùy chọn
#         ]
#     },
#     # TIME: "tháng 12", "ngày 20 tháng 11", "năm 2025", "tháng sau"
#     {
#         "label": "TIME",
#         "pattern": [
#             {"TEXT": {"IN": ["tháng", "ngày"]}},
#             {"TEXT": {"REGEX": r"\d{1,2}"}}  # Số tháng/ngày
#         ]
#     },
#     {
#         "label": "TIME",
#         "pattern": [
#             {"TEXT": "năm"},
#             {"TEXT": {"REGEX": r"\d{4}"}}  # Số năm
#         ]
#     },
#     {
#         "label": "TIME",
#         "pattern": [
#             {"TEXT": "tháng"},
#             {"TEXT": {"IN": ["tới", "sau"]}}  # "tháng tới", "tháng sau"
#         ]
#     },
#     {
#         "label": "TIME",
#         "pattern": [
#             {"TEXT": {"REGEX": r"ngày\s*\d{1,2}/\d{1,2}(?:/\d{4})?"}}  # "ngày 20/11", "ngày 15/12/2025"
#         ]
#     }
# ]

# # Thêm mẫu vào entity_ruler
# ruler.add_patterns(patterns)

# # Thử nghiệm
# text = "Lê Thị Hồng muốn đặt tour đi Phú Quốc vào 12/5 giá tầm 3 triệu đồng"
# doc = nlp(text)
# for ent in doc.ents:
#     print(f"Text: {ent.text}, Label: {ent.label_}")

# try:
#     # os.makedirs(os.path.dirname("extended_intent_train_data.json"), exist_ok=True)
#     with open("ner_train_data.json", "w", encoding="utf-8") as f:
#         json.dump(extended_data, f, ensure_ascii=False, indent=2)
#     print(f"Đã tạo dataset mới với {len(extended_data)} mẫu")
# except Exception as e:
#     print(f"Lỗi khi lưu file: {e}")
# import re

# def extract_price_time(text):
#     # Regex cho giá
#     price_pattern = r'(\d+\.?\d*)\s*(triệu|nghìn|đồng|VND)'
#     prices = re.findall(price_pattern, text)
    
#     # Regex cho thời gian
#     time_pattern = r'(tháng\s*\d+|ngày\s*\d+/\d+|năm\s*\d{4}|\b(?:thứ|chủ\s*nhật)\b)'
#     times = re.findall(time_pattern, text)
    
#     return {"prices": prices, "times": times}

# text = "tôi đang tìm tour có giá trên 10 triệu cuối tuần này"
# result = extract_price_time(text)
# print(result)
# Output: {'prices': [('2', 'triệu')], 'times': ['tháng 6', 'năm 2023']}

# import py_vncorenlp
# import os
# # Automatically download VnCoreNLP components from the original repository
# # and save them in some local machine folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# py_vncorenlp.download_model(save_dir=os.path.join(current_dir, 'VnCoreNLP-1.2'))

# # Load the word and sentence segmentation component
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.join(current_dir, 'VnCoreNLP-1.2'))

# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# output = rdrsegmenter.word_segment(text)

# print(output)
# ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']