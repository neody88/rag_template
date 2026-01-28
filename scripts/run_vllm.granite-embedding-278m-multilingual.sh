export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve ./models/ibm-granite/granite-embedding-278m-multilingual \
  --served-model-name granite-embedding-278m-multilingual \
  --dtype float16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --enable-log-requests \
  --port 8889
