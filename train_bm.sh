echo "start train $(date)"
start_time=$(date +%s)
python fireredasr_llm/train.py --config-path ./conf --config-name salm-qwen2-7b_fc_fc_train.bm.yaml
end_time=$(date +%s)
echo "end train $(date)"
echo "train time: $((end_time - start_time)) seconds"