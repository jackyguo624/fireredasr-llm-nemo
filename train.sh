echo "start train $(date)"
start_time=$(date +%s)
python fireredasr_llm/train.py
end_time=$(date +%s)
echo "end train $(date)"
echo "train time: $((end_time - start_time)) seconds"