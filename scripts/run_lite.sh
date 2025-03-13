# set ENV variables
. scripts/env/set_env.sh
. scripts/env/local_var_lite.sh
export PYTHONPATH=$PYTHONPATH:$(pwd)

result_path='outputs_data/gpt-4o/results_0313/test'
echo $result_path
mkdir -p $result_path

python auto_search_main.py \
    --model 'azure/gpt-4o' \
    --output_folder $result_path/location \
    --eval_n_limit 300 \
    --num_processes 50 \
    --localize \
    --use_function_calling \
    --merge