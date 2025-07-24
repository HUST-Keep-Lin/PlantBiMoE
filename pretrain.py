import os
import torch
import argparse
from datasets import config as hf_config
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from plantbimoe.modeling_plantbimoe import PlantbimoeForMaskedLM
from plantbimoe.configuration_plantbimoe import PlantbimoeConfig
from plantbimoe.tokenization_plantbimoe import PlantbimoeTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_config.HF_DATASETS_CACHE = "./datasets"

# -------- Argument Parsing Function --------
def parse_args():
    parser = argparse.ArgumentParser(description="Train PlantBiMoE with Masked Language Modeling")

    # path parameters
    parser.add_argument("--train_path", type=str, default="data/train.txt", help="Path to training text file")
    parser.add_argument("--valid_path", type=str, default="data/valid.txt", help="Path to validation text file")
    parser.add_argument("--config_path", type=str, default="./config.json", help="Model config path")
    parser.add_argument("--output_dir", type=str, default="./output/pretrain", help="Directory to save model outputs")
    parser.add_argument("--max_length", type=int, default=32770, help="Maximum input sequence length")

    # training parameters
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=8e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.9)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "constant"])
    
    # Training precision and special strategies
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision")
   
    return parser.parse_args()


# -------- Main --------
def main():
    args = parse_args()

    # Tokenizer
    tokenizer = PlantbimoeTokenizer(model_max_length=args.max_length)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    # MLM masking
    def mask_tokens(input_ids, tokenizer):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = tokenizer.mask_token_id
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=7, high=tokenizer.vocab_size, size=labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        return {"input_ids": input_ids, "labels": labels}

    def tokenize_function(examples, tokenizer):
        input_ids = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )["input_ids"]
        return mask_tokens(input_ids, tokenizer)

    def load_and_tokenize(path):
        dataset = load_dataset("text", data_files={"data": path}, split="data")
        return dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, num_proc=4)

    # Data loading
    train_dataset = load_and_tokenize(args.train_path)
    valid_dataset = load_and_tokenize(args.valid_path)

    # Model loading
    config = PlantbimoeConfig.from_pretrained(args.config_path)
    model = PlantbimoeForMaskedLM(config=config)

    # Training parameter settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=20,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        ddp_find_unused_parameters=True,
        save_safetensors=False,
        bf16=args.bf16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()