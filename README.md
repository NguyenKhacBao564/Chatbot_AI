
# Dự án Chatbot AI
## Tổng quan
Dự án xây dựng chatbot tư vấn du lịch tại Việt Nam, sử dụng dữ liệu Q&A để huấn luyện và hỗ trợ người dùng qua giao diện tiếng Việt.
# Cấu trúc thư mục

Chatbot_AI/
│
├── README.md                   ← hướng dẫn cài đặt & chạy MVP end-to-end
├── requirements.txt            ← pip dependencies chung (python-dotenv, faiss-cpu…)
├── .env.example                ← mẫu biến môi trường (GOOGLE_API_KEY,…)
│
├── data/                       ← **Data Pipeline Lead** phụ trách
│   ├── raw/                    ← file gốc chưa xử lý
│   │   └── faq.json
│   ├── processed/              ← sản phẩm sau preprocess
│   │   └── faq_cleaned.json
│   ├── schema_faq.jsonschema   ← JSON-Schema kiểm định cấu trúc
│   └── notebooks/              ← Jupyter khám phá / thống kê Q&A
│
├── scripts/                    ← script CLI 1-lần hoặc cron
│   ├── preprocess_faq.py       ← chuẩn hoá + validate ⇒ /data/processed
│   ├── wikivoyage_crawl.py     ← (tùy chọn) crawler nguồn mở
│   └── make_dataset.sh         ← shell gộp crawl → preprocess
│
├── retrieval/                  ← **Retrieval & RAG Engineer**
│   ├── index_faq.py            ← build Faiss index từ faq_cleaned.json
│   ├── retrieval.py            ← hàm `get_retrieved_context()`
│   ├── faq_index.bin           ← file Faiss (git-ignored lớn)
│   ├── faq_metadata.json       ← map id → Q&A
│   └── tests/
│       └── test_retrieval.py
│
├── llm/                        ← **AI Integration Engineer**
│   ├── llm_client.py           ← wrapper Gemini (retry, timeout, logging)
│   ├── prompts.py              ← FAQ & fallback template
│   └── tests/
│       └── test_llm_client.py
│
├── evaluation/                 ← **Quality & Performance Engineer**
│   ├── test_questions.json     ← 20 câu hỏi mẫu (in_faq / near_faq / out_faq)
│   ├── test_chatbot.py         ← chạy end-to-end ⇒ test_results.json
│   ├── test_parameters.py      ← grid-search temperature / max_tokens
│   ├── measure_performance.py  ← log latency ⇒ performance_log.csv
│   └── results/
│       ├── test_results.json
│       ├── parameter_test_results.json
│       └── performance_log.csv
│
├── logs/
│   └── chatbot_log.csv         ← ghi request/response thật (rotate sau production)
│
├── app/                        ← phần **service/API UI** (có thể Flask hoặc FastAPI)
│   ├── api/
│   │   └── main.py             ← endpoint /chat ask→retrieve→LLM→json
│   └── ui/
│       └── streamlit_app.py    ← giao diện demo MVP
│
├── docker/
│   ├── Dockerfile              ← build image chạy API + retrieval + llm
│   └── docker-compose.yml      ← local stack (app + optional PG / redis)
│
├── .github/
│   └── workflows/
│       └── ci.yml              ← lint + pytest khi push PR
│
└── docs/                       ← tài liệu nội bộ
    ├── architecture.png
    ├── data_pipeline.md
    ├── retrieval_design.md
    └── api_spec.md


## Yêu cầu

Python 3.12+
Cài đặt các thư viện trong requirements.txt:pip install -r requirements.txt



## Hướng dẫn sử dụng

### Làm sạch dữ liệu:
python scripts/preprocess_faq.py --input data/raw/faq_group1.json --output data/processed/faq_group1_cleaned.json

### Lặp lại cho các file khác (faq_group2.json, faq_group3.json, v.v.).

### Gộp dữ liệu:
#python scripts/mergeData.py

### Kết quả lưu tại data/processed/faq_combined.json.

Tích hợp vào chatbot: Sử dụng faq_combined.json trong pipeline RAG để huấn luyện chatbot.


## hi chú

Đảm bảo file .env có biến GOOGLE_API_KEY để sử dụng API trong dataFromcallAPI.py.
Kiểm tra dữ liệu trong faq_combined.json trước khi tích hợp để tránh lỗi định dạng.

