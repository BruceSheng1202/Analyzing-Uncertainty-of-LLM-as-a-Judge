# LLM Judge Uncertainty Analysis

## Project Overview

This repository is the old version of code repository for my paper "[Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction](https://arxiv.org/abs/2509.18658)" and please to see [new version in another repository](https://github.com/BruceSheng1202/Analyzing_Uncertainty_of_LLM-as-a-Judge). This repository contains code, data, and experiments for analyzing and comparing conformal prediction methods to quantify the uncertainty of large language models acting as “judges” on rating-based tasks. We implement and evaluate Boosted CP, CHR, CQR, LVD, Ordinal APS/RC, and R2CCP across multiple datasets.

## Repository Structure

All files and data live at the root of this repository:

```text
.
├── boosted-conformal/         # Boosted CP method scripts and outputs
├── calsize/                   # Calibration-size experiments and data
├── chr/                       # CHR method scripts and outputs
├── data_results/              # Processed data result files
├── human_performance/         # Human baseline performance data
├── interval_results/          # Generated interval results
├── judge/                     # Judge prompt and output logs
├── LVD/                       # LVD method scripts and outputs
├── midpoints/                 # Midpoint comparison experiments
├── model_logits/              # Raw model logits
├── model_paths/               # Model checkpoint paths
├── raw_scores/                # Raw prediction scores
├── reprompt/                  # Reprompt experiment notebooks and data
├── BoostedCP_random.ipynb     # Boosted CP random-split notebook
├── CHR_random.ipynb           # CHR random-split notebook
├── CQR_random.ipynb           # CQR random-split notebook
├── LVD_random.ipynb           # LVD random-split notebook
├── OrdinalAPS_random.ipynb    # Ordinal APS random-split notebook
├── OrdinalRC_random.ipynb     # Ordinal RC random-split notebook
├── R2CCP_random.ipynb         # R2CCP random-split notebook
├── R2CCP_calsize.ipynb        # R2CCP calibration-size notebook
├── calsize_plot.ipynb         # Calibration-size plotting notebook
├── heteroskedasticity_ht.ipynb# Heteroskedasticity testing notebook
├── plot_instances.ipynb       # Instance-level result plotting notebook
├── score_performance.ipynb    # Aggregated score performance notebook
├── R2CCP-0.0.8-py3-none-any.whl# R2CCP wheel for local install
└── README.md                  # Project overview and instructions
```

