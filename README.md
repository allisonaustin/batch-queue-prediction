## Batch-Scheduled HEP Job Prediction

A benchmarking and evaluation suite for evaluating tree-based models and deep tabular architectures on High-Performance Computing (HPC) batch job execution logs. This repository provides end-to-end pipelines for model training, threshold selection, feature importance extraction, and protocol evaluation under both random and temporal data split schemes. We also include analyses of submit-time vs. match-time feature matrices, hold reason codes, and exit code/signal evaluation.

### Prediction Tasks

1. Job Failure (classification) at match-time (**E1**)
2. Queue wait time (regression) at submit-time (**E2**)
3. Fault attribution (classification) at match-time (**E3**)

### Supported Models

* XGBoost, LightGBM, CatBoost, MLP
* TabNet
* TabR
* FT-Transformer
* SAINT
* Hierarchical NN
* TSMixer

### Repo Structure

```
fife-batch-jobs/
├── scripts/
│   ├── vis/						# Data explorer visualization
│   ├── eval/
│   │   ├── harness.py              # Main CLI evaluation harness
│   │   └── helper.py               # Model loaders, metrics, and prediction routines
│   ├── train/                      # Model implementations
│   ├── output/                     # Evaluation metrics and feature importance output JSONs
│   ├── data-explorer.ipynb  		# Job data visualization app example (work in progress)
│   ├── data-analysis.ipynb  		# Initial job data analysis
	├── feat-engineering.ipynb  	# Training and testing setup for job prediction tasks
	└── pred-analysis.ipynb         # Job prediction result visualizations
├── .gitignore                      # Dataset and runtime configuration
├── CLAUDE.md                       # Global instructions for Claude            
├── FIFE-Docs.md                    # FIFE Batch Queue data documentation            
├── MODELING.md                     # Detailed modeling design notes
├── README.md                       # Repo info
└── config.json						# Dataset and runtime configuration
```

### Requirements

#### Dependencies

Install the required packages:

```
pip install torch numpy pandas scikit-learn xgboost lightgbm catboost pytorch-tabnet matplotlib seaborn
```

#### Dataset(s)

TBD

##### Structure

- `Xmatch.npy` & `Xsub.npy`: feature matrices (match-time and submit-time features)
- `failed.npy`: finary failure target labels
- `wait_sv.npy`: raw queue wait times in seconds
- `tr_mask.npy` & `te_mask.npy`: train and test split marks (`tr` for random and `te` for temporal)
- `targets_and_masks.npz`: aggregated dataset dictionary

### How to run

All training and evaluation tasks are managed through the CLI in `scripts/eval/harness.py`:

```
python scripts/eval/harness.py <experiment> <model> [split]
```

#### Arguments

- `<experiment>`:
  - `e1` -- run job failure classification
  - `e2` -- run queue wait time regression
  - `e3` -- run fault attribution classification
- `<model>`: `xgb`, `lgb`, `cat`, `mlp`, `tabnet`, `saint`, `ft`, `tsmixer`, `tabr`, `hierarchical`
- `[split]`: specify training split protocol (optional) `random`, `temporal`, or `both` (default)

#### Usage

1. Job failure classification using XGBoost with temporal split:

```
python scripts/eval/harness.py e1 xgboost both
```

### Metrics

##### Job failure classification/fault attribution

Precision, recall, F1, ROC-AUC, PR-AUC

##### Queue wait time

- log1p MAE: mean absolute error evaluated on $log(1 + wait\_time)$
- raw MAE (s): ean absolute error converted back to raw seconds
- sMAPE (%): symmetric mean absolute percentage error (bounded between $0\%$ and $200\%$)
- within-2x ratio: proportion of predictions falling within a factor of 2 of actual wait times

### References

Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS 30), 3146–3154.

Prokhorenkova, L., et al. (2018). *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS 31), 6638–6648.

Arik, S. O., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687. [https://arxiv.org/abs/1908.07442](https://arxiv.org/abs/1908.07442)

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS 34), 18932–18943. [https://arxiv.org/abs/2106.11959](https://arxiv.org/abs/2106.11959)

Somepalli, G., Goldblum, M., Suri, A., Geiping, J., & Goldstein, T. (2021). *SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-training*. arXiv preprint arXiv:2106.01342. [https://arxiv.org/abs/2106.01342](https://arxiv.org/abs/2106.01342)

Chen, S. A., Li, C. L., Yoder, N., Sercu, T., & Pfister, T. (2023). *TSMixer: Lightweight MLP-Mixer for Multivariate Time Series Forecasting*. arXiv preprint arXiv:2303.06053. [https://arxiv.org/abs/2303.06053](https://arxiv.org/abs/2303.06053)

Lovell, A., et al. (2024). *A Hierarchical Deep Learning Approach for Predicting Job Queue Times in HPC Systems*. IEEE/ACM International Conference for High Performance Computing, Networking, Storage and Analysis Workshops (SC24 Workshops).
