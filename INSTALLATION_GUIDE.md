# LocAgent Installation Guide

This guide helps you set up LocAgent with proper dependency management, addressing common installation issues.

## 🚨 Common Installation Issues

The original `requirements.txt` contains CUDA dependencies that may not be compatible with:
- Python 3.12
- Windows systems
- Systems without NVIDIA GPUs

## 🔧 Solution Options

### Option 1: Recommended Setup (Python 3.11)

```bash
# 1. Create new conda environment with Python 3.11
conda deactivate
conda remove -n locagent --all -y  # Remove existing if needed
conda create -n locagent python=3.11 -y
conda activate locagent

# 2. Use the fixed requirements file
pip install -r requirements_fixed.txt

# 3. Test installation
python setup_environment.py
```

### Option 2: Automated Setup Script

```bash
# Activate your environment first
conda activate locagent

# Run the setup script
python setup_environment.py
```

### Option 3: Manual Core Installation

If you encounter issues, install core dependencies manually:

```bash
# Core ML/AI libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install datasets huggingface-hub transformers
pip install litellm openai azure-identity

# Code analysis libraries  
pip install tree-sitter tree-sitter-languages
pip install networkx matplotlib pandas numpy
pip install bm25s nltk

# Development tools
pip install ipython jupyter
```

## 🧪 Testing Your Installation

After installation, test that everything works:

```bash
# Test core imports
python -c "import datasets; import litellm; import networkx; import tree_sitter; print('✅ Core dependencies working!')"

# Test the main script
python auto_search_main.py --help

# Test with a simple example (requires API key)
export OPENAI_API_KEY="your-key-here"
python auto_search_main.py --dataset 'czlll/SWE-bench_Lite' --split 'test' --eval_n_limit 1 --model 'openai/gpt-4o-2024-05-13' --localize --output_folder ./test_output
```

## 🔑 Environment Variables

Set up your API keys:

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional

# For Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="..."

# For graph indexing (optional)
export GRAPH_INDEX_DIR="./indexes/graph_index_v2.3"
export BM25_INDEX_DIR="./indexes/BM25_index"
```

## 🚀 Quick Start After Installation

1. **Basic localization run:**
```bash
python auto_search_main.py \
    --dataset 'czlll/SWE-bench_Lite' \
    --split 'test' \
    --model 'openai/gpt-4o-2024-05-13' \
    --localize \
    --merge \
    --output_folder ./results \
    --eval_n_limit 5 \
    --use_function_calling
```

2. **Build dependency graphs (optional but recommended):**
```bash
python dependency_graph/batch_build_graph.py \
    --dataset 'czlll/SWE-bench_Lite' \
    --split 'test' \
    --num_processes 10 \
    --download_repo
```

## 📊 What Each Component Does

- **`auto_search_main.py`** - Main entry point for code localization
- **`dependency_graph/`** - Builds code dependency graphs for better search
- **`plugins/location_tools/`** - Search and retrieval tools
- **`repo_index/`** - Repository indexing and context management
- **`evaluation/`** - Evaluation metrics and benchmarking

## 🐛 Troubleshooting

### "No module named 'tree_sitter'"
```bash
pip install tree-sitter tree-sitter-languages
```

### "CUDA not available" warnings
This is normal if you don't have an NVIDIA GPU. The system will use CPU-only PyTorch.

### "litellm" import errors
```bash
pip install --upgrade litellm
```

### Memory issues during graph building
Reduce the number of processes:
```bash
python dependency_graph/batch_build_graph.py --num_processes 5
```

## 📝 Notes

- The system works with CPU-only PyTorch (no GPU required)
- Graph indexing is optional but improves performance
- You can start with a small dataset (`--eval_n_limit 5`) to test
- Results are saved as JSONL files in the output folder

## 🔗 Useful Commands

```bash
# Check what's installed
pip list | grep -E "(torch|datasets|litellm|tree-sitter)"

# Test specific components
python -c "from plugins.location_tools import LocationToolsRequirement; print('Plugins working!')"
python -c "from dependency_graph.build_graph import build_graph; print('Graph building working!')"
