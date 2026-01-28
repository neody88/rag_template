export CUDA_VISIBLE_DEVICES=0
vllm serve ./models/ibm-granite/granite-4.0-micro \
  --served-model-name granite-4.0-micro \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --dtype float16 \
  --max-model-len 1024 \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests \
  --port 8889
