# set api key
. scripts/env/set_env.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/graph_index'
export BM25_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/BM25_index'

result_path='YOUR_OUTPUT_PATH'
echo $result_path
mkdir -p $result_path

python auto_search_main.py \
    --dataset 'czlll/SWE-bench_Lite' \
    --split 'test' \
    --model 'azure/gpt-4o' \
    --output_folder $result_path/location \
    --eval_n_limit 300 \
    --num_processes 50 \
    --localize \
    --use_function_calling \
    --merge \
    --simple_desc

# python auto_search_main.py \
#     --dataset 'czlll/Loc-Bench' \
#     --split 'test' \
#     --model 'azure/gpt-4o' \
#     --output_folder $result_path/location \
#     --eval_n_limit 660 \
#     --num_processes 50 \
#     --localize \
#     --use_function_calling \
#     --merge \
#     --simple_desc