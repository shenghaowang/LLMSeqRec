# LLMSeqRec

This repository provides a PyTorch Lightning implementation of a sequential recommendation model enhanced with pretrained language model (LLM) embeddings. The core architecture is based on [Kang & McAuley, 2018](https://arxiv.org/abs/1808.09781).

---

## 🚀 Features

- ✅ Simplied SASRec implementation using PyTorch Lightning
- ✅ LLM-enhanced item embeddings (e.g., MiniLM)
- ✅ Support for MovieLens 1M and Amazon datasets (Beauty & Video Games)
- ✅ Hit Rate and NDCG evaluation at top-k
- ✅ YAML-based configuration and reproducible training pipeline

---

## 📦 Installation

To set up the project using a Python virtual environment, follow the steps below.

1. **Clone the repository**
```bash
git clone https://github.com/shenghaowang/LLMSeqRec.git
cd LLMSeqRec
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the `pre-commit` hooks**
```bash
pre-commit install
pre-commit run --all-files
```

---

## 📂 Datasets

Create a `raw` directory at the root of the project and download the MovieLens 1M, Amazon Video Games, and Amazon Beauty review datasets into this folder. You can obtain the datasets from the official sources linked below. Refer to [`src/config/config.yaml`](src/config/config.yaml) for the expected file paths and filenames.

* [MovieLens 1M](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset)
* [Amazon reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)

---

## 🔥 Usage

### Preprocess user-item interaction data and precompute metadata embeddings

```bash
export PYTHONPATH=src
python src/preprocess/main.py dataset=<dataset_name>
```

Supported dataset names:

* `ml-1m`: MovieLens 1M
* `amzn_games`: Amazon Video Games
* `amzn_beauty`: Amazon Beauty

### Train and evaluate recommendation model

```bash
export PYTHONPATH=src
python src/train_and_eval/main.py dataset=<dataset_name> model=<model_name>
```

Supported models:

* `poprec`: PopRec
* `mf`: Matrix Factorization
* `sasrec`: SASRec
* `llmseqrec`: LLMSeqRec

---

## References
* Kang, W.-C., & McAuley, J. (2018). Self-attentive sequential recommendation. arXiv. https://arxiv.org/abs/1808.09781
