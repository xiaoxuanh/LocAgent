export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configure git to trust the repositories
git config --global --add safe.directory '*'

python build_bm25_index.py \
        --dataset 'princeton-nlp/SWE-bench_Lite_bm25_13K' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 12 \
        --download_repo