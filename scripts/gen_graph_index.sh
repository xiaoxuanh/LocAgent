export PYTHONPATH=$PYTHONPATH:$(pwd)
BASE_DIR='YOUR_REPO_DIR'
GRAPH_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/graph_index'

cd ./graph_encoder
python dependency_graph/batch_build_graph.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --repo_path $BASE_DIR/playground/build_graph \
        --out_path $GRAPH_INDEX_DIR \
        --num_processes 50 \
        --download_repo


# python dependency_graph/batch_build_graph.py \
#         --dataset 'czlll/Loc-Bench' \
#         --split 'test' \
#         --repo_path $BASE_DIR/playground/build_graph \
#         --out_path $GRAPH_INDEX_DIR \
#         --num_processes 50 \
#         --download_repo