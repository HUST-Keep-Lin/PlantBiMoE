# PlantBiMoE

## Environment
To get started, create a conda environment
```bash
conda create -n plantbimoe python=3.8
conda activate plantbimoe
```
Install dependencies
```bash
git clone https://github.com/HUST-Keep-Lin/PlantBiMoE.git
cd PlantBiMoE
pip install -r requirements
```


## Pretrain
Launch pretraining run using the command line
```bash
accelerate launch pretrain.py \
    --train_path "data/train.txt" \
    --valid_path "data/valid.txt" \
    --config_path "./config.json" \
    --output_dir "./output/pretrain" \
    --max_length 32770 \
    --num_epochs 20 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 8e-3 \
    --weight_decay 0.1 \
    --adam_beta1 0.95 \
    --adam_beta2 0.9 \
    --warmup_steps 1000 \
    --accumulation_steps 8 \
    --lr_scheduler "cosine" \
    --bf16
```
## Finetune
Launch fine-tune using PlantBiMoE, you should download PlantBiMoE from [huggingface](https://huggingface.co/plant-llms/PlantBiMoE).
```bash
accelerate launch finetune.py \
    --model_name_or_path "path/to/model" \
    --train_path "data/train.tsv" \
    --valid_path "data/valid.tsv" \
    --test_path "data/test.tsv" \
    --config_path "./config.json" \
    --output_dir "./output/finetune" \
    --max_length 512 \
    --num_labels 2 \
    --num_epochs 10 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --bf16

```