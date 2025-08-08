import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import argparse
import subprocess

# Setup basic logging to see the progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_hf_model(
    llm_dir: str,
    firered_checkpoint_path: str,
    output_dir: str,
):
    """
    Merges FireRedASR LoRA weights with the base LLM.
    """

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the base Qwen2 HuggingFace model.
    # We load it onto the CPU in bfloat16 to manage memory.
    logging.info(f"Loading base LLM from: {llm_dir}")
    model = AutoModelForCausalLM.from_pretrained(llm_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    
    # Also load the tokenizer, which is needed for the final conversion.
    tokenizer = AutoTokenizer.from_pretrained(llm_dir)
    
    # 2. Define and apply the same LoRA configuration used during training.
    # This prepares the model architecture to accept the LoRA weights.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "gate_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    logging.info("Applying LoRA configuration to the base model.")
    model = get_peft_model(model, lora_config)

    # 3. Load the FireRedASR checkpoint.
    logging.info(f"Loading FireRedASR checkpoint from: {firered_checkpoint_path}")
    checkpoint = torch.load(firered_checkpoint_path, map_location="cpu", weights_only=False)
    llm_state_dict = checkpoint['model_state_dict']
    
    # 4. Isolate and remap the LoRA weights from the checkpoint.
    lora_state_dict = {}
    for key, value in llm_state_dict.items():
        if key.startswith("llm."):
            # The checkpoint saves the PEFT model under the key 'llm'.
            # To load it into our model, we just need to remove this prefix.
            # e.g., 'llm.base_model...' becomes 'base_model...'
            new_key = key.replace("llm.", "")
            lora_state_dict[new_key] = value

    # 5. Load the remapped weights into the PEFT model.
    # We use strict=False because the checkpoint only contains LoRA weights.
    # This correctly ignores the base model weights which are not in the checkpoint.
    incompatible_keys = model.load_state_dict(lora_state_dict, strict=False)
    logging.info(f"LoRA weight loading report: {incompatible_keys}")
    logging.info("Successfully loaded LoRA weights into the model.")

    # 6. Merge the loaded LoRA weights into the base model's weights.
    logging.info("Merging LoRA weights...")
    model = model.merge_and_unload()
    logging.info("LoRA weights merged successfully.")

    # 7. Save the merged model in HuggingFace format.
    # This is an intermediate step before NeMo conversion.
    logging.info(f"Saving merged HuggingFace model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Saved tokenizer files to: {output_dir}")


if __name__ == '__main__':
    # --- Configuration ---
    argparse = argparse.ArgumentParser(description="Merge FireRedASR LoRA weights with Qwen2.")
    argparse.add_argument('--llm_dir', type=str, required=True, help='Path to qwen2.5-7b instruct dir')
    argparse.add_argument('--firered_checkpoint', type=str, required=True, help='Path to the FireRedASR checkpoint')
    argparse.add_argument('--output_dir', type=str, required=True, help='Directory to save the merged model')
    args = argparse.parse_args()
    llm_dir, firered_checkpoint, output_dir = args.llm_dir, args.firered_checkpoint, args.output_dir
    merge_hf_model(
        llm_dir=llm_dir,
        firered_checkpoint_path=firered_checkpoint,
        output_dir=output_dir,
    ) 
