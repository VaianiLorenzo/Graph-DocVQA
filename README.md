# 📄 ADJ-GAT: Adaptive Graph Attention Networks for Document Representation

Welcome to **ADJ-GAT**, a research-focused Python project that explores adaptive graph-based learning techniques for representing and reasoning over **visually rich documents (VRDs)**.

This repository implements a pipeline that:

- Loads OCR and visual embeddings for documents,
- Constructs document-level graphs based on embedding similarity,
- Applies **Graph Attention Networks (GATs)** to extract meaningful representations,
- Supports experimentation with loss functions like **Binary Cross Entropy (BCE)** and **Focal Loss**.

> 💡 This project is especially useful for researchers and practitioners working on **key information extraction**, **document classification**, or **graph-based deep learning**.

---

## 🧠 Key Features

- ✅ Modular code for loading document metadata and embeddings.
- 🔍 Customizable graph construction via top-k similarity.
- 🧱 Plug-and-play architecture for experimenting with different loss functions.
- ⚙️ Easy to adapt to new document datasets with OCR and embeddings.

---

## 📁 Repository Structure

adjgat/\
├── adjgat.ipynb # Main Jupyter notebook (preprocessing, graph construction, training loop)\
├── utils/ # Optional: Helper functions (graph building, metrics, loss)\
├── models/ # Optional: GAT and auxiliary models\
└── README.md # You're reading this!\


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/adjgat.git
cd adjgat
```

### 2. Set up your environment

We recommend using conda:

```bash
conda create -n adjgat python=3.9
conda activate adjgat
pip install -r requirements.txt
```

### 3. Prepare the data
Update the notebook with the paths to your local .pkl files for:
- train_doc_info.pkl / val_doc_info.pkl
- train_E5Vembeddings.pkl / val_E5Vembeddings.pkl
- Any question embeddings (if used)

These files should contain:
- OCR metadata (doc_info)
- E5V embeddings (visual/textual embeddings)
- Optional: QA-style prompt embeddings

### 4. Run the notebook
```bash
jupyter notebook adjgat.ipynb
```

### 🧪 Example Use Cases
- 📑 Document Classification using structural and visual features.
- 🔍 Key Information Extraction (KIE) with graph reasoning.
- 🔬 Ablation studies on similarity thresholds and graph connectivity.

### ⚙️ Configuration Options
Inside the notebook:
```python
top_k_smilarity = 5         # Number of neighbors to connect in the graph
loss_set = 'bce'            # Choose between 'bce' or 'focal'
```

### ✨ Acknowledgements
Inspired by recent work in Document AI and Graph Neural Networks (GNNs).
Utilizes powerful libraries like PyTorch, tqdm, and pickle.
