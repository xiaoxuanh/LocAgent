#!/bin/bash
# filepath: /home/azureuser/cloudfiles/code/Users/v-xhou/locagent/LocAgent/scripts/run_azure.sh

# Get access token for Cognitive Services
echo "Getting Azure access token..."
tokenJson=$(az account get-access-token --scope https://cognitiveservices.azure.com/.default --output json)
if [ $? -eq 0 ]; then
    echo "✅ Token obtained successfully"
else
    echo "❌ Failed to get access token"
    exit 1
fi

# Extract access token from JSON
accessToken=$(echo "$tokenJson" | jq -r '.accessToken')

# Set Azure OpenAI environment variables
export AZURE_OPENAI_ENDPOINT="https://mass-swc.openai.azure.com/"
export AZURE_OPENAI_API_KEY="$accessToken"
export AZURE_OPENAI_API_VERSION="2025-04-14"
export AZURE_OPENAI_DEPLOYMENT="gpt-4.1"

# Clear any conflicting OpenAI variables
unset OPENAI_API_KEY
unset OPENAI_API_BASE

# Verify environment variables are set
echo "=== Environment Variables Check ==="
echo "AZURE_OPENAI_ENDPOINT: $AZURE_OPENAI_ENDPOINT"
echo "AZURE_OPENAI_API_VERSION: $AZURE_OPENAI_API_VERSION"
echo "AZURE_OPENAI_DEPLOYMENT: $AZURE_OPENAI_DEPLOYMENT"
if [ -n "$AZURE_OPENAI_API_KEY" ]; then
    echo "AZURE_OPENAI_API_KEY: ${AZURE_OPENAI_API_KEY:0:20}..."
else
    echo "AZURE_OPENAI_API_KEY: NOT SET"
fi
echo "=== End Check ==="

curl -X POST "https://mass-swc.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_OPENAI_API_KEY" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export GRAPH_INDEX_DIR='index_data/SWE-bench_Lite_bm25_13K/graph_index_v2.3'
export BM25_INDEX_DIR='index_data/SWE-bench_Lite_bm25_13K/BM25_index'

# Set rate limiting environment variables
export LITELLM_RATE_LIMIT_RETRY="true"
export LITELLM_REQUEST_TIMEOUT="30"

result_path='./results'
echo "$result_path"
mkdir -p "$result_path"

# Activate conda environment first
source venv/bin/activate

python auto_search_main.py \
    --dataset 'princeton-nlp/SWE-bench_Lite_bm25_13K' \
    --split 'test' \
    --model 'azure/gpt-4.1-mini' \
    --localize \
    --merge \
    --output_folder "$result_path/GPT4-1" \
    --eval_n_limit 300 \
    --num_processes 1 \
    --use_function_calling \
    --simple_desc \
    --num_samples 1