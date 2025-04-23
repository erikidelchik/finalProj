import torch
from datasets import load_dataset
from transformers import (
    CLIPProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Step 1: Load the dataset from folders ===
dataset = load_dataset("imagefolder", data_dir="dataset/train")

# === Step 2: Train/Validation split ===
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = split["train"]
val_ds = split["test"]

# === Step 3: Load CLIP processor and model ===
MODEL_ID = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# === Step 4: Preprocessing function ===
def preprocess(example):
    inputs = processor(images=example["image"], return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"][0],
        "labels": example["label"]
    }

train_ds = train_ds.map(preprocess, remove_columns=["image"])
val_ds = val_ds.map(preprocess, remove_columns=["image"])

# === Step 5: Training arguments ===
training_args = TrainingArguments(
    output_dir="./clip_sensitive_classifier",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    learning_rate=2e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available()
    # Uncomment below if you upgrade `transformers` to 4.20+:
    # evaluation_strategy="epoch",
    # save_strategy="epoch"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # get index of highest logit
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# === Step 6: Trainer setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# === Step 7: Train ===
trainer.train()

# === Step 8: Manual evaluation (if no evaluation_strategy above) ===
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

model.save_pretrained("clip_sensitive_classifier")
processor.save_pretrained("clip_sensitive_classifier")