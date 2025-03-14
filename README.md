# LocAgent: Graph-Guided LLM Agents for Code Localization

<p align="center">
   📖&nbsp; <a href="https://arxiv.org/abs/2503.09089" target="_blank">Paper</a>
   | 📊&nbsp; <a href="https://huggingface.co/datasets/czlll/Loc-Bench" target="_blank">Loc-bench</a>
   | 🤗&nbsp; <a href="https://huggingface.co/czlll/Qwen2.5-Coder-7B-CL" target="_blank">Qwen2.5-Coder-7B-CL</a>
   | 🤗&nbsp; <a href="https://huggingface.co/czlll/Qwen2.5-Coder-32B-CL" target="_blank">Qwen2.5-Coder-32B-CL</a>

</p>

<!-- We welcome contributions from everyone to help improve and expand auto-search-agent. This document outlines the process for contributing to the project. -->

<!-- ## Table of Contents
1. [Environment Setup](#environment-setup) -->

## Setup
1. Follow these steps to set up your development environment:
   ```
   git clone git@github.com:gersteinlab/LocAgent.git
   cd LocAgent

   conda create -n locagent python=3.12
   conda activate locagent
   pip install -r requirements.txt
   ```

2. Set `api_key` in `scripts/env/set_env.sh`
   ```
   # such as OPENAI API key
   export OPENAI_API_KEY="sk-123..."
   export OPENAI_API_BASE="https://XXXXX"
   ```

## Launch LocAgent
- Parse the codebase for each issue in the benchmarck with `scripts/gen_graph_index.sh` to get graph indexes.
   you can use `--dataset` and `--split` to select the benchmark (by default it will be SWE-Bench Lite)
   After parsing, export the directory of the graph indexes such as:
   ```
   export GRAPH_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/graph_index'

   # bm25 index will be generted during the localization process.
   export BM25_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/BM25_index'
   ```
- Run the script `scripts/run_lite.sh` to lauch LocAgent.


## Cite Us

   ```
  @article{chen2025locagent,
  title={LocAgent: Graph-Guided LLM Agents for Code Localization},
  author={Chen, Zhaoling and Tang, Xiangru and Deng, Gangda and Wu, Fang and Wu, Jialong and Jiang, Zhiwei and Prasanna, Viktor and Cohan, Arman and Wang, Xingyao},
  journal={arXiv preprint arXiv:2503.09089},
  year={2025}
  }
   ```
