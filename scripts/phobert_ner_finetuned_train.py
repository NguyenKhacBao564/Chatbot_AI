# # Huấn luyện mô hình NER với PhoBERT

# import json
# from transformers import Trainer, TrainingArguments

# def read_json(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     sentences = [item["sentence"].split() for item in data]  # Chia câu thành danh sách từ
#     labels = [item["labels"] for item in data]
#     return sentences, labels

# # Đọc dữ liệu
# train_sentences, train_labels = read_json("ner_train_data.json")


# # Lấy danh sách nhãn duy nhất
# all_labels = set()
# for labels in train_labels:
#     all_labels.update(labels)
# label_list = sorted(list(all_labels))  # Ví dụ: ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']
# label2id = {label: i for i, label in enumerate(label_list)}
# id2label = {i: label for i, label in enumerate(label_list)}

# from transformers import AutoTokenizer
# import torch

# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# def align_labels_with_tokens(sentences, labels, max_length=128):
#     tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

#     for words, word_labels in zip(sentences, labels):
#         # Tokenize từng từ và căn chỉnh nhãn
#         token_ids = []
#         aligned_labels = []

#         for word, label in zip(words, word_labels):
#             tokenized_word = tokenizer.encode(word, add_special_tokens=False)
#             token_ids.extend(tokenized_word)
#             aligned_labels.extend([label2id[label]] + [-100] * (len(tokenized_word) - 1))

#         # Thêm [CLS] và [SEP]
#         token_ids = [tokenizer.cls_token_id] + token_ids[:max_length-2] + [tokenizer.sep_token_id]
#         aligned_labels = [-100] + aligned_labels[:max_length-2] + [-100]

#         # Padding
#         padding_length = max_length - len(token_ids)
#         token_ids += [tokenizer.pad_token_id] * padding_length
#         aligned_labels += [-100] * padding_length
#         attention_mask = [1] * (len(token_ids) - padding_length) + [0] * padding_length

#         tokenized_inputs["input_ids"].append(token_ids)
#         tokenized_inputs["attention_mask"].append(attention_mask)
#         tokenized_inputs["labels"].append(aligned_labels)

#     # Chuyển thành tensor
#     tokenized_inputs["input_ids"] = torch.tensor(tokenized_inputs["input_ids"])
#     tokenized_inputs["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"])
#     tokenized_inputs["labels"] = torch.tensor(tokenized_inputs["labels"])

#     return tokenized_inputs

# # Tiền xử lý dữ liệu
# data = align_labels_with_tokens(train_sentences, train_labels)

# from torch.utils.data import Dataset

# class NERDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         return {key: val[idx] for key, val in self.encodings.items()}

#     def __len__(self):
#         return len(self.encodings["input_ids"])

# dataset = NERDataset(data)


# from transformers import AutoModelForTokenClassification

# model = AutoModelForTokenClassification.from_pretrained(
#     "vinai/phobert-base",
#     num_labels=len(label_list),
#     id2label=id2label,
#     label2id=label2id
# )



# training_args = TrainingArguments(
#     output_dir="./phobert_ner_finetuned",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
# )


# import numpy as np
# from evaluate import load
# from torch.utils.data import random_split

# train_size = int(0.8 * len(dataset))
# eval_size = len(dataset) - train_size
# train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
# metric = load("seqeval")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)

#     true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
#     true_predictions = [
#         [id2label[p] for p, l in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]

#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# # Lưu mô hình
# model.save_pretrained("./phobert_ner_finetuned")
# tokenizer.save_pretrained("./phobert_ner_finetuned")

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from torch.utils.data import Dataset, random_split
from evaluate import load

# Hàm đọc dữ liệu JSON
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Kiểm tra định dạng
    for i, sentence in enumerate(data):
        if not isinstance(sentence, list) or not all(isinstance(token, dict) and "word" in token and "label" in token for token in sentence):
            raise ValueError(f"Dữ liệu JSON không đúng định dạng tại câu {i}: {sentence}")
    
    sentences = [[token["word"] for token in sentence] for sentence in data]
    labels = [[token["label"] for token in sentence] for sentence in data]
    
    # Kiểm tra số từ và nhãn
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        if len(sentence) != len(label):
            raise ValueError(f"Số từ ({len(sentence)}) không khớp với số nhãn ({len(label)}) trong câu {i}: {sentence}")
    
    return sentences, labels

# Đọc dữ liệu
train_sentences, train_labels = read_json("ner_train_data2.json")

# Tạo danh sách nhãn
all_labels = set()
for labels in train_labels:
    all_labels.update(labels)
label_list = sorted(list(all_labels))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
print("Nhãn:", label_list)

# Khởi tạo tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Hàm token hóa và căn chỉnh nhãn
def align_labels_with_tokens(sentences, labels, max_length=128):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for words, word_labels in zip(sentences, labels):
        token_ids = []
        aligned_labels = []

        for word, label in zip(words, word_labels):
            tokenized_word = tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(tokenized_word)
            aligned_labels.extend([label2id[label]] + [-100] * (len(tokenized_word) - 1))

        # Thêm [CLS] và [SEP]
        token_ids = [tokenizer.cls_token_id] + token_ids[:max_length-2] + [tokenizer.sep_token_id]
        aligned_labels = [-100] + aligned_labels[:max_length-2] + [-100]

        # Padding
        padding_length = max_length - len(token_ids)
        token_ids += [tokenizer.pad_token_id] * padding_length
        aligned_labels += [-100] * padding_length
        attention_mask = [1] * (len(token_ids) - padding_length) + [0] * padding_length

        tokenized_inputs["input_ids"].append(token_ids)
        tokenized_inputs["attention_mask"].append(attention_mask)
        tokenized_inputs["labels"].append(aligned_labels)

    # Chuyển thành tensor
    tokenized_inputs["input_ids"] = torch.tensor(tokenized_inputs["input_ids"])
    tokenized_inputs["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"])
    tokenized_inputs["labels"] = torch.tensor(tokenized_inputs["labels"])

    return tokenized_inputs

# Tiền xử lý dữ liệu
data = align_labels_with_tokens(train_sentences, train_labels)

# Tạo dataset
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

dataset = NERDataset(data)

# Chia tập train/eval
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# Khởi tạo mô hình
model = AutoModelForTokenClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Hàm tính metric
metric = load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./phobert_ner_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    no_cuda=True if not torch.cuda.is_available() else False,
    dataloader_pin_memory=False
)

# Khởi tạo DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True, max_length=128)

# Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Huấn luyện
trainer.train()

# Lưu mô hình
model.save_pretrained("./phobert_ner_finetuned")
tokenizer.save_pretrained("./phobert_ner_finetuned")

# Kiểm tra mô hình
from transformers import pipeline

ner_pipeline = pipeline("ner", model="./phobert_ner_finetuned", tokenizer=tokenizer, aggregation_strategy="simple")
result = ner_pipeline("Nguyễn Văn An muốn đi tour tại Sài Gòn")
print(result)



# from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
# from datasets import Dataset
# from pyvi import ViTokenizer
# import json
# import os
# # from seqeval.metrics import f1_score, precision_score, recall_score
# from vncorenlp import VnCoreNLP
# # Tải tokenizer và mô hình PhoBERT
# model_name = "vinai/phobert-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# jar_path = os.path.join(current_dir, "VnCoreNLP-1.1.1.jar")
# vncorenlp = VnCoreNLP(jar_path, annotators="wseg,pos,ner", max_heap_size='-Xmx2g')
# # Định nghĩa nhãn theo chuẩn BIO
# label_list = ["O", "B-LOCATION", "I-LOCATION", "B-TIME", "I-TIME", "B-PRICE", "I-PRICE"]
# label2id = {label: idx for idx, label in enumerate(label_list)}
# id2label = {idx: label for idx, label in enumerate(label_list)}

# # Tải và xử lý dữ liệu
# with open("ner_train_data_with_price.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# def prepare_data(data):
#     tokenized_inputs = []
#     labels = []
#     for item in data:
#         query = vncorenlp.tokenize(item["query"])
#         encoding = tokenizer(query, return_offsets_mapping=True, padding="max_length", truncation=True, max_length=128)
#         labels_item = ["O"] * len(encoding["input_ids"])
#         for entity in item["entities"]:
#             start, end = entity["start"], entity["end"]
#             label = entity["label"]
#             entity_text = entity["text"]
#             first_token = True
#             for i in range(len(encoding["offset_mapping"])):
#                 token_start, token_end = encoding["offset_mapping"][i]
#                 token = tokenizer.convert_ids_to_tokens(encoding["input_ids"][i])[0].replace("▁", " ")
#                 if token_start >= start and token_end <= end and token_end != 0:
#                     if first_token and token in entity_text:
#                         labels_item[i] = f"B-{label}"
#                         first_token = False
#                     elif not first_token and token in entity_text:
#                         labels_item[i] = f"I-{label}"
#         tokenized_inputs.append(encoding)
#         labels.append([label2id[label] for label in labels_item])
#     return tokenized_inputs, labels

# tokenized_inputs, labels = prepare_data(data)
# dataset = Dataset.from_dict({
#     "input_ids": [x["input_ids"] for x in tokenized_inputs],
#     "attention_mask": [x["attention_mask"] for x in tokenized_inputs],
#     "labels": labels
# })

# # Chia tập dữ liệu
# train_test_split = dataset.train_test_split(test_size=0.2)
# train_dataset = train_test_split["train"]
# eval_dataset = train_test_split["test"]

# # Hàm xử lý dữ liệu
# data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# # Thiết lập tham số huấn luyện
# training_args = TrainingArguments(
#     output_dir="./phobert_ner_finetuned",
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     eval_strategy="epoch",
#     save_steps=500,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# # Khởi tạo trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     # compute_metrics=compute_metrics,
# )

# # Huấn luyện mô hình
# trainer.train()

# # Lưu mô hình
# model.save_pretrained("./phobert_ner_finetuned")
# tokenizer.save_pretrained("./phobert_ner_finetuned")

# print("Huấn luyện hoàn tất!")