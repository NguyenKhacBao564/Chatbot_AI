# install: pip install google-generativeai
import os
import json
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
import re
from jsonschema import validate, ValidationError


# Load schema để kiểm tra dữ liệu
SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 5, "maxLength": 300},
        "answer": {"type": "string", "minLength": 5, "maxLength": 500},
        "tags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["question", "answer", "tags"],
    "additionalProperties": False
}

load_dotenv()
# 1. Thiết lập API Key (đảm bảo bạn đã đăng ký và lấy được API key free tier của Google Generative AI)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 2. Danh sách các chủ đề và tag tương ứng
# Nhóm TOPICS (chọn nhóm để sinh dữ liệu)
TOPICS_GROUPS = {
    "group1": [
        {"tag": "destination_basics", "name": "Thông tin cơ bản về điểm đến"},
        {"tag": "weather_best_time", "name": "Thời tiết và thời điểm tốt nhất"},
        {"tag": "transport", "name": "Phương tiện di chuyển"},
        {"tag": "scenery_things_to_do", "name": "Cảnh quan và hoạt động"},
        {"tag": "visa_entry", "name": "Visa và yêu cầu nhập cảnh"}
    ],
    "group2": [
        {"tag": "tour_booking", "name": "Đặt tour du lịch"},
        {"tag": "pricing_payment_currency", "name": "Giá cả, thanh toán và tiền tệ"},
        {"tag": "promotions_membership", "name": "Khuyến mãi và thành viên"},
        {"tag": "accommodation", "name": "Chỗ ở khi du lịch"},
        {"tag": "food_dining", "name": "Ẩm thực và ăn uống"}
    ],
    "group3": [
        {"tag": "health_safety_insurance", "name": "Sức khỏe, an toàn và bảo hiểm"},
        {"tag": "culture_etiquette", "name": "Văn hóa và phong tục"},
        {"tag": "events_festivals", "name": "Sự kiện và lễ hội"},
        {"tag": "connectivity_sim_wifi", "name": "Kết nối, SIM và Wi-Fi"},
        {"tag": "budgeting_tips", "name": "Mẹo tiết kiệm chi phí"}
    ],
    "group4": [
        {"tag": "accessibility_family", "name": "Du lịch gia đình và người khuyết tật"},
        {"tag": "solo_female_safety", "name": "An toàn cho nữ du lịch một mình"},
        {"tag": "sustainability_eco", "name": "Du lịch bền vững và sinh thái"},
        {"tag": "emergency_laws", "name": "Khẩn cấp và luật pháp địa phương"},
        {"tag": "packing_checklist", "name": "Danh sách chuẩn bị hành lý"}
    ]
}

# 3. Prompt mẫu để generate Q&A
PROMPT_TEMPLATE = """
Bạn là một chuyên gia du lịch tại Việt Nam, hỗ trợ người dùng qua chatbot tiếng Việt. 
Tôi cần bạn tạo dữ liệu Q&A để huấn luyện chatbot tư vấn du lịch.  
Hãy tạo **10 cặp câu hỏi và câu trả lời** về chủ đề: “{topic_name}”.  
**Yêu cầu:**  
- Câu hỏi và câu trả lời phải bằng **tiếng Việt**, ngắn gọn, tự nhiên, và thân thiện như chatbot trả lời người dùng.  
- Câu hỏi dài 1-2 câu, không quá 300 ký tự.  
- Tránh lặp lại nội dung giữa các cặp Q&A.  
- Đáp ứng định dạng JSON: [{{"question": "...", "answer": "..."}}].  
- Câu trả lời dài 2-3 câu, không quá 500 ký tự, phải cung cấp thông tin chi tiết, hữu ích và có chiều sâu (ví dụ: thêm mẹo, số liệu cụ thể, hoặc gợi ý thực tế).
- Ví dụ: [{{"question": "Làm sao để đến Đà Nẵng?", "answer": "Bạn có thể đi máy bay đến sân bay Đà Nẵng hoặc đi tàu từ Hà Nội. Chuyến bay mất khoảng 1 giờ 30 phút và giá vé từ 1 triệu đồng. Taxi từ sân bay về trung tâm khoảng 70.000 đồng. Hoặc bạn có thể đi phương tiện công cộng với gía rẻ hơn."}}]
"""

class GeminiClient:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

    async def generate_faq_for_topic(self, topic_name, retries=3):
        prompt = PROMPT_TEMPLATE.format(topic_name=topic_name)
        for attempt in range(1, retries + 1):
            try:
                response = await self.model.generate_content_async(
                    contents=prompt,
                    generation_config={"temperature": 0.7, "max_output_tokens": 1200}
                )
                text = response.text
                # print(f"Raw response for {topic_name}: {text}")  # Log phản hồi thô
                try:
                    data = json.loads(text)
                    return data
                except json.JSONDecodeError:
                    m = re.search(r"\[.*\]", text, re.DOTALL)
                    if m:
                        return json.loads(m.group(0))
                    raise ValueError("Invalid JSON format in response")
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg or "server error" in error_msg:
                    if attempt < retries:
                        print(f"Attempt {attempt} failed: {error_msg}. Retrying after {attempt} seconds...")
                        await asyncio.sleep(attempt)
                        continue
                print(f"Error generating FAQ for {topic_name}: {error_msg}")
                return []
        print(f"Max retries reached for {topic_name}. Skipping...")
        return []
        # async def generate_faq_for_topic(self, topic_name, retries=3):
        #     prompt = PROMPT_TEMPLATE.format(topic_name=topic_name)
        #     for attempt in range(1, retries + 1):
        #         try:
        #             response = await self.model.generate_content_async(
        #                 contents=prompt,
        #                 generation_config={"temperature": 0.7, "max_output_tokens": 800}
        #             )
        #             text = response.text
        #             print(f"Raw response for {topic_name}: {text}")  # Log phản hồi thô
        #             try:
        #                 data = json.loads(text)
        #                 return data
        #             except json.JSONDecodeError:
        #                 m = re.search(r"\[.*\]", text, re.DOTALL)
        #                 if m:
        #                     return json.loads(m.group(0))
        #                 raise ValueError("Invalid JSON format in response")
        #         except Exception as e:
        #             error_msg = str(e).lower()
        #             if "rate limit" in error_msg or "quota" in error_msg or "server error" in error_msg:
        #                 if attempt < retries:
        #                     print(f"Attempt {attempt} failed: {error_msg}. Retrying after {attempt} seconds...")
        #                     await asyncio.sleep(attempt)
        #                     continue
        #             print(f"Error generating FAQ for {topic_name}: {e}")
        #             return []
        #     print(f"Max retries reached for {topic_name}. Skipping...")
        #     return []
        
# 4. Hàm kiểm tra và lọc các cặp Q&A hợp lệ        
def validate_qa_pairs(qa_pairs):
    """Kiểm tra và lọc các cặp Q&A hợp lệ"""
    valid_pairs = []
    for qa in qa_pairs:
        qa_copy = qa.copy()
        qa_copy["tags"] = []  # Tạm thêm tags để validate
        try:
            validate(qa_copy, SCHEMA)
            if len(qa["question"]) > 5 and len(qa["answer"]) > 5:  # Tránh rỗng hoặc quá ngắn
                valid_pairs.append(qa)
        except ValidationError as ve:
            print(f"Invalid QA pair skipped: {qa}. Error: {ve.message}")
            continue
    return valid_pairs

async def main(group_name):
    if group_name not in TOPICS_GROUPS:
        print(f"Group {group_name} not found. Available groups: {list(TOPICS_GROUPS.keys())}")
        return

    client = GeminiClient()
    all_faq = []
    topics = TOPICS_GROUPS[group_name]

    for topic in topics:
        print(f"Generating FAQ for topic: {topic['name']}")
        qa_pairs = await client.generate_faq_for_topic(topic["name"])
        qa_pairs = validate_qa_pairs(qa_pairs)
        for qa in qa_pairs:
            qa["tags"] = [topic["tag"]]
            all_faq.append(qa)
        print(f"Generated {len(qa_pairs)} valid Q&A for {topic['name']}")

    # Lưu ra file
    output_file = f"/Users/nguyen_bao/Documents/PTIT/Junior_2/ltw/Chatbot_AI/data/raw/faq_{group_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_faq, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo {len(all_faq)} mục Q&A vào {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate FAQ data for a specific group of topics.")
    parser.add_argument("--group", required=True, help="Group name (e.g., group1, group2, group3, group4)")
    args = parser.parse_args()
    asyncio.run(main(args.group))