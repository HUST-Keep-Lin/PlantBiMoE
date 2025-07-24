import os
import torch
import argparse
import numpy as np
from datasets import config as hf_config
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from plantbimoe.modeling_plantbimoe import PlantbimoeForSequenceClassification
from plantbimoe.configuration_plantbimoe import PlantbimoeConfig
from plantbimoe.tokenization_plantbimoe import PlantbimoeTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_config.HF_DATASETS_CACHE = "./datasets"

# -------- Argument Parsing Function --------
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune PlantBiMoE for Sequence Classification")

    # path parameters
    parser.add_argument("--model_name_or_path", type=str, default="path/to/model", help="Path to pretrained model")
    parser.add_argument("--train_path", type=str, default="data/train.tsv", help="Path to training TSV file")
    parser.add_argument("--valid_path", type=str, default="data/valid.tsv", help="Path to validation TSV file")
    parser.add_argument("--test_path", type=str, default="data/test.tsv", help="Path to test TSV file")
    parser.add_argument("--config_path", type=str, default="./config.json", help="Model config path")
    parser.add_argument("--output_dir", type=str, default="./output/finetune", help="Directory to save model outputs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input sequence length")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels")

    # training parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    # Training precision and special strategies
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")
   
    return parser.parse_args()


# -------- Metrics Calculation Function --------
def compute_metrics(eval_pred):
    """Calculate evaluation metrics for classification tasks"""
    predictions, labels = eval_pred
    # Get predicted classes
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    # Calculate F1 score (macro average)
    f1 = f1_score(labels, predictions, average="macro")
    
    return {"accuracy": accuracy, "f1_macro": f1}


# -------- Main --------
def main():
    args = parse_args()

    # Tokenizer
    tokenizer = PlantbimoeTokenizer(model_max_length=args.max_length)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Data loading and preprocessing
    def tokenize_function(examples):
        """Convert text to model input format"""
        encoding = tokenizer(
            examples["sequence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        encoding["labels"] = examples["label"]
        return encoding

    def load_and_process(path):
        """Load TSV data and perform preprocessing"""
        dataset = load_dataset("csv", data_files={"data": path}, delimiter="\t", split="data")
        return dataset.map(tokenize_function, batched=True, num_proc=4)

    # Load datasets
    train_dataset = load_and_process(args.train_path)
    valid_dataset = load_and_process(args.valid_path)


    model = PlantbimoeForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Training parameter settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    
    # Save the best model
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()