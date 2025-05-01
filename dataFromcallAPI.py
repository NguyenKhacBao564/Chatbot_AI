# install: pip install google-generativeai
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
# 1. Thiết lập API Key (đảm bảo bạn đã đăng ký và lấy được API key free tier của Google Generative AI)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 2. Danh sách các chủ đề và tag tương ứng
TOPICS = [
    {"tag": "destination", "name": "Địa điểm du lịch"},
    {"tag": "weather",     "name": "Thời tiết và hoạt động phù hợp"},
    {"tag": "transport",   "name": "Phương tiện di chuyển"},
    {"tag": "scenery",     "name": "Cảnh quan và trải nghiệm"},
]

# 3. Prompt mẫu để generate Q&A
PROMPT_TEMPLATE = """
Bạn là một chuyên gia du lịch. 
Tôi đang cần bạn tạo ra một danh sách các câu hỏi và câu trả lời ngắn gọn về một chủ đề du lịch do hiện tại tôi đang tạo dữ liệu cho việc train một mô hình chatbot AI cho mục đích tư vấn du lịch.
Hãy tạo ra 10 cặp câu hỏi và câu trả lời có độ dài vừa đủ (5-, rõ ràng (2-3 câu mỗi câu trả lời) 
về chủ đề: “{topic_name}”. 
Mỗi cặp trả về theo định dạng JSON object:
[{{"question": "...", "answer": "..."}}].
"""

def generate_faq_for_topic(topic_name):
    prompt = PROMPT_TEMPLATE.format(topic_name=topic_name)

    # Dùng Chat API
    resp = genai.chat.completions.create(
      model= 'gemini-2.0-flash',
      temperature=0.7,
      max_output_tokens=800,
      messages=[{"author": "user", "content": prompt}],
    )

    text = resp.candidates[0].message.content

    # parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\[.*\]", text, re.S)
        data = json.loads(m.group(0)) if m else []
    return data

def main():
    all_faq = []
    for topic in TOPICS:
        items = generate_faq_for_topic(topic["name"])
        # Gắn tag
        for qa in items:
            qa["tags"] = [topic["tag"]]
            all_faq.append(qa)
    # Lưu ra file
    with open("/Users/nguyen_bao/Documents/PTIT/Junior_2/ltw/Chatbot_AI/faq.json", "w", encoding="utf-8") as f:
        json.dump(all_faq, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo {len(all_faq)} mục Q&A vào faq.json")

if __name__ == "__main__":
    main()
