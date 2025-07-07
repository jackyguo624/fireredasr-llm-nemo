echo "start validation $(date)"
start_time=$(date +%s)
python fireredasr_llm/validate.py --config-path ./conf --config-name salm-qwen2-7b_fc_fc_valid.bm.yaml
end_time=$(date +%s)
echo "end validation $(date)"
echo "validation time: $((end_time - start_time)) seconds"