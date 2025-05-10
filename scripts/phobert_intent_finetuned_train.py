# from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# from datasets import Dataset
# import json
# import re
# from vncorenlp import VnCoreNLP
# import os
# import torch

# # Kiểm tra GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Thiết bị được sử dụng: {device}")

# # Khởi tạo VnCoreNLP
# current_dir = os.path.dirname(os.path.abspath(__file__))
# jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
# vncorenlp = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx2g')

# # Tải tokenizer và mô hình PhoBERT cơ bản
# model_name = "vinai/phobert-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8).to(device)  # Di chuyển mô hình sang GPU

# # Đọc dữ liệu huấn luyện
# with open("processed_intent_data2.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Hàm tiền xử lý dữ liệu đầu vào 
# def preprocess_text(text):
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Chuẩn bị dữ liệu
# def prepare_data(data, max_length=128):
#     inputs = []
#     labels = []
#     intent_mapping = {
#         "find_tour_with_location": 0,
#         "find_tour_with_time": 1,
#         "find_tour_with_price": 2,
#         "find_tour_with_location_and_time": 3,
#         "find_tour_with_location_and_price": 4,
#         "find_tour_with_time_and_price": 5,
#         "find_with_all": 6,
#         "out_of_scope": 7
#     }

#     for item in data:
#         if "query" not in item or "intent" not in item:
#             print(f"Bỏ qua bản ghi không hợp lệ: {item}")
#             continue
#         query = preprocess_text(item["query"])
#         intent = item["intent"]
        
#         # Phân đoạn từ bằng VnCoreNLP
#         segmented_query = vncorenlp.tokenize(query)
#         if not segmented_query or not segmented_query[0]:
#             print(f"Phân đoạn thất bại cho query: {query}")
#             continue
#         segmented_query = " ".join(word for sentence in segmented_query for word in sentence)
#         inputs.append(segmented_query)
#         labels.append(intent_mapping[intent])

#     tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
#     return tokenized_inputs, labels

# tokenized_inputs, labels = prepare_data(data)
# dataset = Dataset.from_dict({
#     "input_ids": tokenized_inputs["input_ids"],
#     "attention_mask": tokenized_inputs["attention_mask"],
#     "labels": labels
# })

# # Chia tập dữ liệu
# train_test_split = dataset.train_test_split(test_size=0.2)
# train_dataset = train_test_split["train"]
# eval_dataset = train_test_split["test"]

# # Thiết lập tham số huấn luyện
# training_args = TrainingArguments(
#     output_dir="./phobert_intent_finetuned",
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     warmup_steps=100,
#     logging_steps=100,
#     save_total_limit=2,
#     fp16=True if torch.cuda.is_available() else False,  # Sử dụng FP16 để tăng tốc trên GPU
# )

# # Định nghĩa hàm để di chuyển batch sang GPU
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = torch.argmax(torch.tensor(logits), dim=-1)
#     accuracy = (predictions == torch.tensor(labels)).float().mean()
#     return {"accuracy": accuracy.item()}

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=compute_metrics
# )

# # Huấn luyện
# trainer.train()

# # Lưu mô hình và tokenizer
# model.save_pretrained("./phobert_intent_finetuned")
# tokenizer.save_pretrained("./phobert_intent_finetuned")
# print("Huấn luyện hoàn tất!")



#Huấn luyện mô hình PhoBERT cho phân loại ý định

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import re
from vncorenlp import VnCoreNLP
import os
# Khởi tạo VnCoreNLP một lần
current_dir = os.path.dirname(os.path.abspath(__file__))
jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
vncorenlp = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx2g')

# Tải tokenizer và mô hình PhoBERT cơ bản
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)  # 4 ý định

# Đọc dữ liệu huấn luyện
with open("merged_intent_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Hàm tiền xử lý dữ liệu đầu vào 
def preprocess_text(text):
    # Xóa các ký tự đặc biệt, chỉ giữ lại chữ cái, số và dấu cách
    text = re.sub(r'[^\w\s]', '', text)
    # Chuẩn hóa khoảng trắng (loại bỏ khoảng trắng thừa)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Chuẩn bị dữ liệu
def prepare_data(data, max_length=128):
    inputs = []
    labels = []
    intent_mapping = {
        "find_tour_with_location": 0,
        "find_tour_with_time": 1,
        "find_tour_with_price": 2,
        "find_tour_with_location_and_time": 3,
        "find_tour_with_location_and_price": 4,
        "find_tour_with_time_and_price": 5,
        "find_with_all": 6,
        "out_of_scope": 7
    }

    for item in data:
        if "query" not in item or "intent" not in item:
            print(f"Bỏ qua bản ghi không hợp lệ: {item}")
            continue
        query = preprocess_text(item["query"])
        intent = item["intent"]
        
        # Phân đoạn từ bằng VnCoreNLP
        segmented_query = vncorenlp.tokenize(query)
        if not segmented_query or not segmented_query[0]:  # Kiểm tra nếu phân đoạn thất bại
            print(f"Phân đoạn thất bại cho query: {query}")
            continue
        segmented_query = " ".join(word for sentence in segmented_query for word in sentence)
        inputs.append(segmented_query)
        print(f"Segmented query: {segmented_query}")
        labels.append(intent_mapping[intent])

    tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return tokenized_inputs, labels

tokenized_inputs, labels = prepare_data(data)
print("Input IDs:", tokenized_inputs["input_ids"])
print("Attention Mask:", tokenized_inputs["attention_mask"])
dataset = Dataset.from_dict({
    "input_ids": tokenized_inputs["input_ids"],
    "attention_mask": tokenized_inputs["attention_mask"],
    "labels": labels
})

# Chia tập dữ liệu thành train và eval (80% train, 20% eval)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./phobert_intent_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=100,
    save_total_limit=2,
    fp16=True
)

# Huấn luyện
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()

# Lưu mô hình
model.save_pretrained("./phobert_intent_finetuned")
tokenizer.save_pretrained("./phobert_intent_finetuned")
print("Huấn luyện hoàn tất!")


# from vncorenlp import VnCoreNLP
# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
# print("Đang khởi tạo VnCoreNLP...")
# try:
#     vncorenlp = VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx512m')
#     print("Khởi tạo VnCoreNLP hoàn tất!")
#     test_query = "Đây là câu kiểm tra."
#     segmented = vncorenlp.tokenize(test_query)
#     print("Kết quả phân đoạn:", segmented)
# except Exception as e:
#     print(f"Lỗi: {e}")