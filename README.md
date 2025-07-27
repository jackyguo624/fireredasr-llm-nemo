# fireredasr-llm-nemo

This is a nemo example repo for fireredasr-llm,  including finetuning and inference.
By leveraging NeMo, applying SDTP parallelism and ZeRO-1 optimization enables both fine-tuning and inference on consumer-grade GPUs (e.g., RTX 3090/4090), making them feasible hardware choices for large-scale speech models.


# Step up
```bash
## Setup Nemo docker image according to https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html
docker pull nvcr.io/nvidia/nemo:25.04

## git clone fireredasr_llm repo

git clone https://github.com/jackyguo624/fireredasr-llm-nemo.git && cd fireredasr-llm-nemo
```

# Train and validation
```bash
# prepare your dataset in lhotse format first， refer to https://lhotse.readthedocs.io/en/latest/index.html
# For finetune
./train.sh

# For validation
./valid.sh

```


