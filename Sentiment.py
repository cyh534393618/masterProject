import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. 加载数据集并预处理
def load_data(tokenizer, max_length=512, train_size=1000, test_size=500):
    dataset = load_dataset("imdb")

    # 数据预处理：将文本转换为 BERT 可接受的输入格式
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 选择部分数据用于训练和测试，减少训练时间
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(train_size))
    test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(test_size))

    return train_dataset, test_dataset

# 2. 定义评价指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 3. 训练模型
def train_model(model, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer

# 4. 评估模型
def evaluate_model(trainer):
    results = trainer.evaluate()
    print(f"Test Accuracy: {results['eval_accuracy']}")
    print(f"Test F1-Score: {results['eval_f1']}")
    return results

# 5. 测试句子的情感
def predict_sentiment(model, tokenizer, sentence, max_length=512):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment

# 6. 主函数
def main():
    # 加载预训练的 BERT 分词器和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # 加载和预处理数据
    train_dataset, test_dataset = load_data(tokenizer)

    # 训练模型
    trainer = train_model(model, train_dataset, test_dataset)

    # 评估模型
    evaluate_model(trainer)

    # 测试句子的情感
    sentence = "I love this movie so much!"
    sentiment = predict_sentiment(model, tokenizer, sentence)
    print(f"Sentence: '{sentence}' -> Sentiment: {sentiment}")

if __name__ == "__main__":
    main()