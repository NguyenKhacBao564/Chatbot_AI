from fastapi import FastAPI, HTTPException
from main import TourRetrievalPipeline
from pydantic import BaseModel
import uvicorn
# Khởi tạo FastAPI
app = FastAPI()
# Khởi tạo pipeline một lần duy nhất
pipeline = TourRetrievalPipeline()
# Định nghĩa model dữ liệu
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"


@app.post("/chat")
async def handle_query(request: QueryRequest):
    return pipeline.get_tour_response(request.query)

# @app.post("/reset")
# async def reset_session(request: ResetRequest):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     pipeline = TourRetrievalPipeline(
#         index_file=os.path.join(current_dir, "faq_index.faiss"),
#         metadata_file=os.path.join(current_dir, "faq_metadata.json")
#     )
#     pipeline.reset_session(request.user_id)
#     return {"status": "success", "message": f"Đã đặt lại thông tin cho người dùng {request.user_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)