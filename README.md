# Identifying GDPR-Critical Tasks in Business Process Descriptions – An Automatic Approach Trained on a Synthetic Data Set: Replication Package

This replication package contains Jupyter notebooks for training, validating, and explaining transformer-based models 
for detection of processes in the context of GDPR-critical business process requirements.

## Overview

The package includes three main notebooks:

1. **TransformerComparison.ipynb** - Comparison of multiple transformer architectures
2. **ModelValidation.ipynb** - External validation of trained model
3. **XAI.ipynb** - Explainability analysis using Integrated Gradients

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
... 
```

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Data Structure

The expected data structure is:

```
project/
├── data/
│   └── input/
│       ├── ProcessDataset.csv
│       └── ExternalValidationDataset.csv
├── models/
├── logs/
├── notebooks/
│   ├── 1-TransformerComparison.ipynb
│   ├── 2-ModelValidation.ipynb
│   └── 3-XAI.ipynb
├── README.md
└── requirements.txt
```

### Data Format

CSV files should contain at least the following columns:
- `Concatenation`: Text column containing the business process requirement text
- `GDPR-criticality`: Label column (0 for non-critical, 1 for GDPR-critical)

The CSV separator should be `;` and encoding `latin1`.

## Notebook Descriptions

### 1. TransformerComparison.ipynb

**Purpose**: Trains and compares multiple transformer architectures for sequence classification.

**Key Features**:
- Supports multiple pre-trained models (RoBERTa, BERT, DistilBERT, DeBERTa)
- Automated training loop with error handling
- Consistent hyperparameter configuration
- Performance comparison table

**Usage**:
1. Place training data in `../data/input/ProcessDataset.csv`
2. Modify `MODEL_NAMES_LIST` to include desired models
3. Run all cells sequentially
4. Models will be trained and saved to `../models/`

**Output**:
- Trained models saved in `../models/[MODEL_NAME]/`
- Performance comparison CSV in `data/output/transformer_comparison/results_*.csv`
- Console output with training progress and metrics

**Configuration**:
- Training epochs: 3
- Learning rate: 1e-5
- Batch size: 8
- Train/Val/Test split: 60/20/20
- Metric for model selection: Recall

---

### 2. ModelValidation.ipynb

**Purpose**: Validates the best pre-trained transformer model on an external validation dataset.

**Key Features**:
- Loads a fine-tuned RoBERTa model (as best model)
- Evaluates on external validation data
- Generates confusion matrix and classification metrics
- Computes accuracy, recall, and ROC-AUC scores

**Usage**:
1. Ensure a trained model exists in `../models/FacebookAI_roberta-base/`
2. Place validation data in `../data/input/ExternalValidationDataset.csv`
3. Run all cells sequentially
4. Results will be displayed with visualizations

**Output**:
- Confusion matrix visualization
- Classification report with precision, recall, F1-score
- ROC-AUC score
- Recall score

---

### 3. XAI.ipynb

**Purpose**: Performs explainability analysis using Integrated Gradients to identify important words for model predictions.

**Key Features**:
- Integrated Gradients implementation for RoBERTa
- Word-level attribution scores
- Global word importance statistics
- Analysis by prediction outcome (TP, FP, TN, FN)

**Usage**:
1. Ensure a trained model exists in `../models/FacebookAI_roberta-base/`
2. Place test data in `../data/input/ExternalValidationDataset.csv`
3. Run all cells sequentially (may take significant time)
4. Results will be saved to `../results/xai_analysis/`

**Output**:
- `../figures/*`: Most powerful global words for Conf-Mat Metrics for IG w

**Methodology**:

The Integrated Gradients method computes attribution scores by:
1. Creating a baseline (zero embedding)
2. Interpolating from baseline to actual input
3. Computing gradients at interpolation steps
4. Integrating gradients to obtain attribution scores

This provides a principled explanation of which words most influenced the model's decision.

---

## Common Functions

All notebooks share the following standardized functions:

### `set_all_seeds(seed, deterministic=False)`
Sets random seeds for reproducibility across Python, NumPy, PyTorch, and Transformers.

**Parameters**:
- `seed` (int): Random seed value (default: 42)
- `deterministic` (bool): Enable deterministic CUDA operations for full reproducibility

### `log_error(error_message, model_name="Unknown")`
Logs errors to a CSV file with timestamp and model information.

**Parameters**:
- `error_message` (str): Error description
- `model_name` (str): Model identifier

**Output**: Appends to `../logs/error_log.csv`

### `preprocess_function(examples)`
Tokenizes text examples for transformer input.

**Parameters**:
- `examples` (dict): Dictionary with 'text' field

**Returns**: Tokenized inputs with padding and truncation (max_length=512)

### `compute_metrics(eval_pred)`
Computes evaluation metrics from model predictions.

**Parameters**:
- `eval_pred` (tuple): (logits, labels)

**Returns**: Dictionary with 'accuracy' and 'recall' scores

---

## Reproducibility

All notebooks use a fixed random seed (SEED=42) to ensure reproducible results. The `set_all_seeds()` function sets seeds for:
- Python's `random` module
- NumPy's random number generator
- PyTorch's CPU and CUDA random number generators
- Hugging Face Transformers' seed

For maximum reproducibility on GPU, set `deterministic=True`, though this may reduce performance.

---

## Hardware Requirements

**Minimum**:
- CPU: Modern multi-core processor
- RAM: 16 GB
- GPU: Not required but highly recommended

**Recommended**:
- CPU: Intel Xeon or AMD EPYC
- RAM: 32 GB or more
- GPU: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, A5000)
- CUDA: 11.7 or higher

**Notes**:
- XAI.ipynb is computationally intensive and may take hours on CPU
- GPU significantly accelerates training and inference
- TransformerComparison.ipynb trains multiple models sequentially

---

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in `TrainingArguments`
- Use smaller models (e.g., DistilBERT instead of RoBERTa)
- Enable gradient checkpointing
- Use CPU instead of GPU (slower but uses system RAM)

### CUDA Errors
- Ensure CUDA version matches PyTorch installation
- Update GPU drivers
- Check GPU memory usage: `nvidia-smi`

### Import Errors
- Verify all dependencies are installed: `pip list`
- Use a virtual environment to avoid conflicts
- Reinstall transformers: `pip install --upgrade transformers`

### Slow Execution
- Use GPU instead of CPU
- Reduce number of Integrated Gradients steps in XAI.ipynb
- Use smaller validation dataset for faster iterations

---

## Citation

If you use this replication package, please cite:

```
Nake, L., Theuerkauf, R., Sackmann, S. & Kuehnel, S. (2026): 
Identifying GDPR-Critical Tasks in Business Process Descriptions – 
An Automatic Approach Trained on a Synthetic Data Set. 
Business & Information Systems Engineering, forthcoming
```

---

## License
```
MIT
```
---

## Contact

For questions or issues, please contact:
```
Leornard Nake 
@leonard.nake@wiwi.uni-halle.de
Martin-Luther-University Halle-Wittenberg
```

---

## Changelog

### Version 1.0 (2026-01-29)
- Initial release
- Unified and documented three core notebooks
- Standardized common functions with comprehensive docstrings
- Added detailed README with usage instructions

