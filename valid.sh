echo "start validation $(date)"
start_time=$(date +%s)
python fireredasr_llm/validate.py
end_time=$(date +%s)
echo "end validation $(date)"
echo "validation time: $((end_time - start_time)) seconds"