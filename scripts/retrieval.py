import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from scripts.extract_location import extract_info
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn

# app = FastAPI()
# # Định nghĩa model cho request body
# class ChatRequest(BaseModel):
#     query: str


class RetrievalPipeline:
    def __init__(self, index_file="faq_index.faiss", metadata_file="faq_metadata.json"):
        # Tải metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Tải Faiss index
        self.index = faiss.read_index(index_file)

        # Khởi tạo model embedding
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_retrieved_context(self, user_query, top_k=1):
        # Embed câu hỏi
        query_embedding = self.model.encode([user_query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype("float32")

        # Tìm top-k vector gần nhất
        distances, indices = self.index.search(query_embedding, top_k)
        print(f"Distances: {distances}")

        #kiểm tra xem câu trả lời có phù hợp không, tự đặt ngưỡng 
        if distances[0][0] > 1:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn."

        # Tạo context từ metadata
        context_lines = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                item = self.metadata[idx]
                context_lines.append(f"Q: {item['question']}\nA: {item['answer']}")
        print("Context lines :", context_lines)
        return "\n---\n".join(context_lines) if context_lines else ""


# pipeline = RetrievalPipeline()

# # Định nghĩa endpoint
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     try:
#         user_query = request.query
#         context = pipeline.get_retrieved_context(user_query, top_k=1)
#         return {"response": context}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# Ví dụ sử dụng
if __name__ == "__main__":

    # uvicorn.run(app, host="0.0.0.0", port=8000)
    pipeline = RetrievalPipeline()
    client = genai.Client(api_key="AIzaSyADj4p8UequJ8vwSC0JOq43XHkmcJy2TEY")

    while True:
        # Nhập câu hỏi từ người dùng
        user_query = input("Nhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
        
        if user_query.lower() == 'exit':
            break
        
        # Lấy context từ pipeline
        context = pipeline.get_retrieved_context(user_query, top_k=1)
        extracted_info = extract_info(context)
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            config=types.GenerateContentConfig(
                system_instruction="Bạn là một trợ lý ảo tư vấn du lịch thân thiện và chuyên nghiệp. Dựa trên thông tin FAQ được cung cấp trong context, trong context sẽ có dạng Q: (Câu hỏi) và A: (Câu trả lời), từ câu trả lời trong context, tôi muốn bạn trả lời lại câu hỏi của người dùng một cách tự nhiên, ngắn gọn, và đúng trọng tâm. Chỉ sử dụng thông tin từ context, không thêm chi tiết ngoài FAQ, thêm một số từ để trông tự nhiên như con người. Nếu phù hợp, hãy mời người dùng hỏi thêm để nhận hỗ trợ chi tiết hơn."),
            contents=context
        )
        # 
        # In câu trả lời
        print("Câu trả lời sau khi gọi chat bot:", response.text)
        print("===" * 20)