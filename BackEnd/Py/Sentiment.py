import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

MODEL_DIR = "Py/sentiment_saved_model"  # Folder path to save the model

# 1. Load the dataset and preprocess it
def load_data(tokenizer, max_length=512, train_size=1000, test_size=500):
    dataset = load_dataset("imdb")

    # Data preprocessing: convert text into an input format acceptable to BERT
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(train_size))
    test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(test_size))

    return train_dataset, test_dataset

# 2. Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 3. Training model
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

# 4. Evaluation model
def evaluate_model(trainer):
    results = trainer.evaluate()
    print(f"Test Accuracy: {results['eval_accuracy']}")
    print(f"Test F1-Score: {results['eval_f1']}")
    return results

# 5. Test the sentiment of a sentence
def predict_sentiment(model, tokenizer, sentence, max_length=512):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment

# 6. Save model and tokenizer
def save_model_and_tokenizer(model, tokenizer, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

# 7. Load model and tokenizer
def load_model_and_tokenizer(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    # print(f"Model and tokenizer loaded from {model_dir}")
    return model, tokenizer

# 6. main function
def main(sentence):
    # If the model already exists, load the model and tokenizer
    if os.path.exists(MODEL_DIR):
        model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
    else:
        # Otherwise load the pre-trained BERT tokenizer and model and train
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # Load and preprocess data
        train_dataset, test_dataset = load_data(tokenizer)

        # Training model
        trainer = train_model(model, train_dataset, test_dataset)

        # Evaluate the model
        evaluate_model(trainer)

        # Save model and tokenizer
        save_model_and_tokenizer(model, tokenizer, MODEL_DIR)

    # Test the sentiment of a sentence
    # sentence = "I want to make a complaint. I am very dissatisfied with the service yesterday."
    sentiment = predict_sentiment(model, tokenizer, sentence)
    print(f"Sentence: '{sentence}' -> Sentiment: {sentiment}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sentiment Processing Model")

    parser.add_argument('--mode', choices=['test'], required=True, help='Mode of operation')
    parser.add_argument('--sentence', type=str, help='Sentence (for testing)')
    args = parser.parse_args()

    if args.mode == 'test':
        main(args.sentence)