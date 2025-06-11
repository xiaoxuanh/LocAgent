# Get access token for Cognitive Services
Write-Host "Getting Azure access token..."
try {
    $tokenJson = az account get-access-token --scope https://cognitiveservices.azure.com/.default --output json | ConvertFrom-Json
    Write-Host "✅ Token obtained successfully"
} catch {
    Write-Host "❌ Failed to get access token"
    exit 1
}

# Set Azure OpenAI environment variables
$env:AZURE_OPENAI_ENDPOINT="https://mass-swc.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY=$tokenJson.accessToken
$env:AZURE_OPENAI_API_VERSION="2025-01-01-preview"
$env:AZURE_OPENAI_DEPLOYMENT="gpt-4.1-mini"
# Clear any conflicting OpenAI variables
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_API_BASE -ErrorAction SilentlyContinue
# Verify environment variables are set
Write-Host "=== Environment Variables Check ==="
Write-Host "AZURE_OPENAI_ENDPOINT: $env:AZURE_OPENAI_ENDPOINT"
Write-Host "AZURE_OPENAI_API_VERSION: $env:AZURE_OPENAI_API_VERSION"
Write-Host "AZURE_OPENAI_DEPLOYMENT: $env:AZURE_OPENAI_DEPLOYMENT"
if ($env:AZURE_OPENAI_API_KEY) {
    Write-Host "AZURE_OPENAI_API_KEY: $($env:AZURE_OPENAI_API_KEY.Substring(0, 20))..."
} else {
    Write-Host "AZURE_OPENAI_API_KEY: NOT SET"
}
Write-Host "=== End Check ==="

$env:PYTHONPATH=$env:PYTHONPATH + ":" + (Get-Location)
$env:GRAPH_INDEX_DIR='index_data/SWE-bench_Lite_bm25_13K/graph_index_v2.3'
$env:BM25_INDEX_DIR='index_data/SWE-bench_Lite_bm25_13K/BM25_index'

$result_path='./results'
$result_path = './results'
Write-Host $result_path
New-Item -ItemType Directory -Force -Path $result_path

# Activate conda environment first
conda activate locagent

python auto_search_main.py `
    --dataset 'princeton-nlp/SWE-bench_Lite_bm25_13K' `
    --split 'test' `
    --model 'azure/gpt-4.1-mini' `
    --localize `
    --merge `
    --output_folder $result_path/location `
    --eval_n_limit 25 `
    --num_processes 1 `
    --use_function_calling `
    --simple_desc