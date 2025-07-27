import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import subprocess

# Setup basic logging to see the progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_and_convert(
    llm_dir: str,
    firered_checkpoint_path: str,
    output_dir: str,
    nemo_repo_path: str
):
    """
    Merges FireRedASR LoRA weights with the base LLM and converts to .nemo format.
    """
    
    # Ensure output directories exist
    merged_hf_output_dir = os.path.join(output_dir, "merged_hf_model")
    nemo_output_file = os.path.join(output_dir, "firered_qwen2_merged.nemo")
    os.makedirs(merged_hf_output_dir, exist_ok=True)

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
    logging.info(f"Saving merged HuggingFace model to: {merged_hf_output_dir}")
    model.save_pretrained(merged_hf_output_dir)
    tokenizer.save_pretrained(merged_hf_output_dir)
    logging.info(f"Saved tokenizer files to: {merged_hf_output_dir}")

    # 8. Convert the merged HF model to a .nemo checkpoint using the NeMo script.
    logging.info(f"Converting merged model to .nemo format at: {nemo_output_file}")
    conversion_script_path = os.path.join(nemo_repo_path, 'scripts/checkpoint_converters/convert_qwen2_hf_to_nemo.py')
    
    if not os.path.exists(conversion_script_path):
        logging.error(f"NeMo conversion script not found at: {conversion_script_path}")
        return

    # Attempt conversion, trying different tensor parallel sizes if needed.
    cmd = [
        'python', conversion_script_path,
        f'--input_name_or_path {merged_hf_output_dir}',
        f'--output_path {nemo_output_file}',
        f'--precision bf16'
    ]
    print(' '.join(cmd))

    subprocess.run(cmd, check=True)
    logging.info(f"Successfully converted model.")

if __name__ == '__main__':
    # --- Configuration ---
    # Path to the base Qwen2 HuggingFace model directory
    LLM_DIR = "/hpc_stor01/home/jiaqi.guo/tools/github/FireRedASR/pretrained_models/FireRedASR-LLM-L/Qwen2-7B-Instruct"
    
    # Path to the FireRedASR-LLM-L checkpoint file
    FIRERED_CHECKPOINT = "/hpc_stor01/home/jiaqi.guo/tools/github/FireRedASR/pretrained_models/FireRedASR-LLM-L/model.pth.tar"
    
    # Directory to save the output .nemo file and intermediate model
    OUTPUT_DIR = "./converted_models"
    
    # Path to your local NeMo repository checkout
    NEMO_REPO_PATH = "/hpc_stor01/home/jiaqi.guo/tools/github/NeMo"

    merge_and_convert(
        llm_dir=LLM_DIR,
        firered_checkpoint_path=FIRERED_CHECKPOINT,
        output_dir=OUTPUT_DIR,
        nemo_repo_path=NEMO_REPO_PATH,
    ) 