import json
import os
import sys
# json_path: rec0714/speechlm_pred_val_wer_aishell-1_test_cuts_inputs_preds_labels.jsonl

def json2txt(json_path):
    with open(json_path, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    # Sort data by id
    data.sort(key=lambda x: x['id'])
    
    with open(json_path.replace('.jsonl', '.pred'), 'w') as fpred, open(json_path.replace('.jsonl', '.gt'), 'w') as fgt:
        for item in data:
            fpred.write(f"{item['id'].replace('_cut', '')}: {item['pred_text']}\n")
            fgt.write(f"{item['id'].replace('_cut', '')}: {item['text']}\n")

if __name__ == "__main__":
    json_path = sys.argv[1]
    json2txt(json_path)