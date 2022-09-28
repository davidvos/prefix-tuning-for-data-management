# Prefix-tuning for entity matching, error detection and data imputation

Our paper on prefix-tuning for data wrangling tasks is currently in review.

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

You should also create folders for output data, metrics and models.

```
mkdir -p outputs/{models,metrics,data} 
```

### Install the necessary packages 
```
pip install -e transformers/
pip install -r requirements.txt
```

### Setup

Edit the config.yaml file to support WandB logging. If you want to run without WandB, please manually remove all WandB statements in main.py.

```
wandb:
  project_name: '<YOUR WandB PROJECT NAME>'
  entity: '<YOUR WandB ENTITY NAME>'
```

## Run 

To run training, evaluation and testing, execute the following commands. For --finetune-type, either choose 'prefix' or 'fine'. For --task, choose' entity-matching', 'error-detection' or 'data-imputation'. Details on all possible arguments can be found in 'utils/utils.py'.

```
python main.py \
    --data_dir 'data/datasets/entity_matching/structured/iTunes-Amazon' \
    --prefix_size 10 \
    --finetune_type 'prefix' \
    --task 'entity-matching' \
    --n_samples 0 \
    --batch_size 16 \
    --n_epochs 10 \
    --lr 5e-5 \
    --seed 1234
```