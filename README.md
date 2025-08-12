# fireredasr-llm-nemo

This is a nemo example repo for fireredasr-llm,  including finetuning and inference.
By leveraging NeMo, applying SDTP parallelism and ZeRO-1 optimization enables both fine-tuning and inference on consumer-grade GPUs (e.g., RTX 3090/4090), making them feasible hardware choices for large-scale speech models.


# Step up
```bash
## Setup Nemo docker image according to https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html
docker pull nvcr.io/nvidia/nemo:25.04

## docker run the image according to your `cpus` and `gpus`
docker run -it --shm-size=1g --rm -w `pwd` --cpus 64 --gpus 8  nvcr.io/nvidia/nemo:25.04 /bin/bash

## git clone fireredasr_llm repo
git clone https://github.com/jackyguo624/fireredasr-llm-nemo.git && cd fireredasr-llm-nemo

## install requirements
pip install -r requirements.txt

## make the patch
patch /opt/NeMo/nemo/lightning/pytorch/callbacks/model_checkpoint.py patch/model_checkpoint.patch
patch /opt/NeMo/nemo/lightning/io/to_config.py patch/to_config.path
```

# Prepare the fireredasr-llm model
```bash
# download Qwen2-7B-Instruct and FireRedASR-LLM-L from huggingface
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir `pwd`/pretrain_model/Qwen2-7B-Instruct
huggingface-cli download FireRedTeam/FireRedASR-LLM-L --local-dir `pwd`/pretrain_model/FireRedASR-LLM-L

# merge the lora with llm
python fireredasr_llm/scripts/merge_firered_lora.py \
    --llm_dir pretrain_model/Qwen2-7B-Instruct  \
    --firered_checkpoint pretrain_model/FireRedASR-LLM-L/model.pth.tar  \
    --output_dir converted_model/merged_qwen2-7b_instruct
```


# Prepare your train and valid dataset in lhotse format
```bash
# prepare your dataset in lhotse format first， refer to https://lhotse.readthedocs.io/en/latest/index.html
# Take aishell-1 for example
lhotse download aishell export
lhotse prepare aishell export/aishell export/aishell

# Convert to Cuts
for f in train dev test; do
python fireredasr_llm/scripts/prepare_lhotse_cuts.py \
    -r export/aishell/aishell_recordings_${f}.jsonl.gz \
    -s export/aishell/aishell_supervisions_${f}.jsonl.gz \
    -o export/aishell/aishell_cuts_${f}.jsonl.gz
```


# Train and validation
```bash
# For finetune
python fireredasr_llm/train.py 

# For validation
python fireredasr_llm/validate.py
```


