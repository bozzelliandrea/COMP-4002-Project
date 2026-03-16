# COMP-4002 — Machine Learning Research Project

A ready-to-use Jupyter Notebook project for machine-learning research.  
Open `notebooks/main.ipynb` to explore a complete end-to-end ML workflow.

---

## Project Structure

```
COMP-4002-Project/
├── notebooks/
│   └── main.ipynb          # Main research notebook (start here)
├── data/
│   └── README.md           # Instructions for adding your own datasets
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading helpers
│   ├── preprocessing.py    # Train/val/test split & feature scaling
│   └── evaluation.py       # Metrics, confusion matrices, model comparison
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/bozzelliandrea/COMP-4002-Project.git
cd COMP-4002-Project
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Launch Jupyter

```bash
jupyter notebook notebooks/main.ipynb
```

---

## What the notebook covers

| Section | Description |
|---------|-------------|
| 1 | Environment setup — imports, reproducibility seed |
| 2 | Data loading — built-in sklearn datasets or your own CSV |
| 3 | Exploratory Data Analysis — distributions, correlations, pairplot |
| 4 | Preprocessing — stratified train/val/test split, StandardScaler |
| 5 | Model training — Logistic Regression, Random Forest, SVM, k-NN, Gradient Boosting (with 5-fold CV) |
| 6 | Validation evaluation — metrics table + confusion matrices |
| 7 | Final test-set evaluation — classification report for the best model |
| 8 | Feature importance — bar chart for tree-based models |

---

## Using your own dataset

Replace the `load_sklearn_dataset(...)` call in **Section 2** of the notebook with:

```python
X, y = load_csv("../data/your_file.csv", target_column="label")
CLASS_NAMES = ["class_a", "class_b", ...]   # update as needed
```

---

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python | 3.10 |
| notebook | 7.0 |
| numpy | 1.26 |
| pandas | 2.1 |
| matplotlib | 3.8 |
| seaborn | 0.13 |
| scikit-learn | 1.4 |
| scipy | 1.12 |

---

## License

See [LICENSE](LICENSE).