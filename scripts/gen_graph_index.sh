export HOME=/home/azureuser
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configure git to trust the repositories
git config --global --add safe.directory '*'

# generate graph index for SWE-bench_Lite
python dependency_graph/batch_build_graph.py \
        --dataset 'princeton-nlp/SWE-bench_Lite_bm25_13K' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 12 \
        --download_repo

# generate graph index for Loc-Bench
# python dependency_graph/batch_build_graph.py \
#         --dataset 'czlll/Loc-Bench_V1' \
#         --split 'test' \
#         --repo_path playground/build_graph \
#         --num_processes 50 \
#         --download_repo