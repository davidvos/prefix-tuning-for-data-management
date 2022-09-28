# Prefix-tuning for entity matching, error detection and data imputation

Prefix-tuning is based on [this](https://github.com/XiangLi1999/PrefixTuning.git) paper and code by Li et al. The language modelling setup for data wrangling tasks is based on [this](https://github.com/HazyResearch/fm_data_tasks) paper and code by HazyResearch. The implementation of T5 is based on [this](https://github.com/ChainsmokersAI/Prompt-Tuning-on-ToTTo) project by ChainsmokersAI. 
## Instructions

### Clone the repo and download the data.

The data can also be manually downloaded from [here](https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz) and should be extracted in the new 'data' directory.
```
git clone https://github.com/davidvos/prefix-tuning-for-data-management.git
cd prefix-tuning-for-data-management
mkdir data
wget https://fm-data-tasks.s3.us-west-1.amazonaws.com/datasets.tar.gz -P data
tar xvf data/datasets.tar.gz -C data/
```

### Install the necessary packages 
```
pip install -e transformers/
pip install -r requirements.txt
```

### Setup

Edit the config.yaml file to support WandB logging.

```
wandb:
  project_name: '<YOUR WandB PROJECT NAME>'
  entity: '<YOUR WandB ENTITY NAME>'
```
