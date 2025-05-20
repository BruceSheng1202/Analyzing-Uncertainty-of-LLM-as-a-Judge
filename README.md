# LLM Judge Uncertainty Analysis

## 🚀 Project Overview

This repository contains code, data, and experiments for analyzing and comparing conformal prediction methods to quantify the uncertainty of large language models acting as "judges" on rating-based tasks. We implement and evaluate Boosted CP, CHR, CQR, LVD, Ordinal APS/RC, and R2CCP across multiple datasets.

## 📁 Repository Structure

```text
.
├── Analysis/                      # Uncertainty analysis notebooks, data, and results
│   ├── boosted-conformal/         # Boosted CP method scripts and outputs
│   ├── calsize/                   # Calibration-size experiments and data
│   ├── chr/                       # CHR method scripts and outputs
│   ├── data_results/              # Processed data result files
│   ├── human_performance/         # Human baseline performance data
│   ├── interval_results/          # Generated interval results
│   ├── judge/                     # Judge prompt and output files
│   ├── LVD/                       # LVD method scripts and outputs
│   ├── midpoints/                 # Midpoint-based evaluation comparisons
│   ├── model_logits/              # Raw model logits
│   ├── model_paths/               # Model checkpoint paths
│   ├── raw_scores/                # Raw prediction scores
│   ├── reprompt/                  # Reprompt experiment notebooks and data
│   ├── BoostedCP_random.ipynb
│   ├── CHR_random.ipynb
│   ├── CQR_random.ipynb
│   ├── LVD_random.ipynb
│   ├── OrdinalAPS_random.ipynb
│   ├── OrdinalRC_random.ipynb
│   ├── R2CCP_random.ipynb
│   ├── R2CCP_calsize.ipynb
│   ├── calsize_plot.ipynb
│   ├── heteroskedasticity_ht.ipynb
│   ├── plot_instances.ipynb
│   ├── score_performance.ipynb
│   ├── R2CCP-0.0.8-py3-none-any.whl
│   └── README                      # Analysis-specific README
├── midpoints/                     # Midpoint comparison scripts and results
├── reprompt/                      # Reprompt experiment notebooks and data
└── README.md                      # Project overview and instructions
```

## ⚙️ Requirements & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<username>/<repo>.git
   cd <repo>
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib jupyterlab statsmodels
   ```

4. **(Optional) Install R2CCP**

   ```bash
   pip install ./Analysis/R2CCP-0.0.8-py3-none-any.whl
   ```

## 📝 Reproduce Experiments

* **Calibration-size**: `Analysis/R2CCP_calsize.ipynb`, `Analysis/calsize_plot.ipynb`
* **Random splits**: Run `*_random.ipynb` notebooks under `Analysis/`
* **Heteroskedasticity tests**: `Analysis/heteroskedasticity_ht.ipynb`
* **Midpoint evaluations**: See files under `Analysis/midpoints/` and top-level `midpoints/`
* **Reprompt experiments**: Notebooks under `Analysis/reprompt/` and top-level `reprompt/`

## 🤝 Contributing

1. Fork → Branch → Commit → PR
2. Follow PEP 8 and document new modules

## 📜 License

MIT License. See [LICENSE](LICENSE).
