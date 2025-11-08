# Ensemble learning model for APT attack detection in network
## Overview
This project focuses on **detecting Advanced Persistent Threat (APT)** activities in network traffic by **extracting flow-based features** and training deep learning models for classification task.
---

## Feature Extraction with CICFlowMeter
We used **CICFlowMeter** to extract statistical features from raw packet capture files (`.pcap`).

- **Data source**: [Stratosphere Laboratory datasets ](https://www.stratosphereips.org/datasets-malware) 
- **Tool:** [CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter)

You can see some samples about dataset in "sample" folder. There are about 1000 sample.

## Model Overview
The following models are implemented:
- GAN (Generative Adversarial Network): for generating synthetic network flow data (unbalanced dataset).
- ELModel (Ensemble Learning Model): combines of LSTM and Switch Transformer.

## To training:
Run: `python main.py`


