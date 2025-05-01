# LocAgent: Graph-Guided LLM Agents for Code Localization

<p align="center">
   📑&nbsp; <a href="https://arxiv.org/abs/2503.09089" target="_blank">Paper</a>
   | 📊&nbsp; <a href="https://huggingface.co/datasets/czlll/Loc-Bench_V1" target="_blank">Loc-bench</a>
   | 🤗&nbsp; <a href="https://huggingface.co/czlll/Qwen2.5-Coder-7B-CL" target="_blank">Qwen2.5-Coder-7B-CL</a>
   | 🤗&nbsp; <a href="https://huggingface.co/czlll/Qwen2.5-Coder-32B-CL" target="_blank">Qwen2.5-Coder-32B-CL</a>
</p>


## ℹ️ Overview
We introduce **LocAgent**, a framework that addresses code localization through graph-based representation.
By parsing codebases into directed heterogeneous graphs, LocAgent creates a lightweight representation that captures code structures and their dependencies, enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning.
 <!-- <div align="center">
  <img src="./assets/overview.png" alt="Overview" width="800">
</div> -->
![MedAgents Benchmark Overview](assets/overview.png)

## ⚙️ Setup
1. Follow these steps to set up your development environment:
   ```
   git clone git@github.com:gersteinlab/LocAgent.git
   cd LocAgent

   conda create -n locagent python=3.12
   conda activate locagent
   pip install -r requirements.txt
   ```

## 🚀 Launch LocAgent
1. (Optional but recommended) Parse the codebase for each issue in the benchmark to generate graph indexes in batch.
   ```
   python dependency_graph/batch_build_graph.py \
         --dataset 'czlll/Loc-Bench_V1' \
         --split 'test' \
         --num_processes 50 \
         --download_repo
   ```
   - `dataset`: select the benchmark (by default it will be `SWE-Bench_Lite`); you can choose from `['czlll/SWE-bench_Lite', 'czlll/Loc-Bench_V1']`(adapted for code localization) and SWE-bench series datasets like `['princeton-nlp/SWE-bench_Lite', 'princeton-nlp/SWE-bench_Verified', 'princeton-nlp/SWE-bench']`
   - `repo_path`: the directory where you plan to pull or have already pulled the codebase
   - `index_dir`: the base directory where the generated graph index will be saved
   - `download_repo`: whether to download the codebase to `repo_path` before indexing

2. Export the directory of the graph indexes and the BM25 sparse index. If not generated in advance, the graph index will be generated during the localization process.
   ```
   export GRAPH_INDEX_DIR='{INDEX_DIR}/{DATASET_NAME}/graph_index_v2.3'
   export BM25_INDEX_DIR='{INDEX_DIR}/{DATASET_NAME}/BM25_index'
   ```

2. Run the script `scripts/run_lite.sh` to lauch LocAgent.
   ```
   python auto_search_main.py \
      --dataset 'czlll/SWE-bench_Lite' \
      --split 'test' \
      --model 'azure/gpt-4o' \
      --localize \
      --merge \
      --output_folder $result_path/location \
      --eval_n_limit 300 \
      --num_processes 50 \
      --use_function_calling \
      --simple_desc
   ```
   - `localize`: set to start the localization process
   - `merge`: merge the result of multiple samples
   - `use_function_calling`: enable function calling features of LLMs. If disabled, codeact will be used to support function calling
   -  `simple_desc`: use simplified function descriptions due to certain LLM limitations. Set to False for better performance when using Claude.

3. Evaluation
   After localization, the results will be saved in a JSONL file. You can evaluate them using `evaluation.eval_metric.evaluate_results`. Refer to `evaluation/run_evaluation.ipynb` for a demonstration.


## 📑 Cite Us

   ```
  @article{chen2025locagent,
  title={LocAgent: Graph-Guided LLM Agents for Code Localization},
  author={Chen, Zhaoling and Tang, Xiangru and Deng, Gangda and Wu, Fang and Wu, Jialong and Jiang, Zhiwei and Prasanna, Viktor and Cohan, Arman and Wang, Xingyao},
  journal={arXiv preprint arXiv:2503.09089},
  year={2025}
  }
   ```
