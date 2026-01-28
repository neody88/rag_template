export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_DISABLE_XET=1

#MODEL_NAME="ibm-granite/granite-4.0-micro"
MODEL_NAME="ibm-granite/granite-embedding-278m-multilingual"
huggingface-cli download $MODEL_NAME --local-dir ./models/$MODEL_NAME
