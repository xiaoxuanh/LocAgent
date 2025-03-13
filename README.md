# LocAgent: Graph-Guided LLM Agents for Code Localization

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

2. Set the environment variable in the script:
   - add `openai_api_key` in `scripts/env/set_env.sh`
   ```
   # OpenAI API key
   export OPENAI_API_KEY="sk-123..."
   export OPENAI_API_BASE="https://XXXXX"
   ```

   - set the index cache directories
   ```
   export GRAPH_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/dependency_graph_index'
   export BM25_INDEX_DIR='{CACHE_DIR}/{DATASET_NAME}/BM25_index'
   ```

## Launch LocAgent
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
