import argparse
import torch
from pathlib import Path
from dataset import PrefixDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import logging
import json

from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

from utils.utils import compute_metrics, setup_logger, parse_args
from prefix_model import PrefixTuning

logger = logging.getLogger(__name__)

args = parse_args()

setup_logger(args.output_dir)
logger.info(json.dumps(vars(args), indent=4))

test_file = "test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Pre-Trained GPT-2 Model
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = tokenizer.eos_token_id).to(device)
# Resize PLM's Embedding Layer
model.resize_token_embeddings(len(tokenizer))
# Freeze LM
for param in model.parameters():
    param.requires_grad=False

prefix_model = PrefixTuning(model.config) 

prefix_model = prefix_model.to(device)

data_dir = data_dir = str(Path(args.data_dir).resolve())

dataset_train = PrefixDataset(tokenizer, data_dir, 'train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)
print(len(dataloader_train))

dataset_test = PrefixDataset(tokenizer, data_dir, 'test')
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=dataset_test.collate_fn)

optimizer = AdamW(model.parameters(), lr=args.lr)

for epoch in range(args.n_epochs):

    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

    prefix_model.train()

    for step, (_, batch, _) in enumerate(dataloader_train):

        batch = batch.to(device)

        prefix = prefix_model(batch_size=batch.shape[0], device=device)
        outputs = model(batch, labels=batch, past_key_values=prefix)
        
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

    with torch.no_grad():

        save_data = {}
        save_data['model_inputs'] = []
        save_data['preds'] = []
        save_data['gts'] = []

        prefix_model.eval()

        for step, (description, _, target) in enumerate(dataloader_test):

            description = description.to(device)
            target = target.to(device)

            outputs = model.generate(description, max_length=100, do_sample=True)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                prediction = output_text.split('answer: ')[1]
            except:
                prediction = ''
                
            description = tokenizer.batch_decode(description, skip_special_tokens=True)[0]
            ground_truth = tokenizer.batch_decode(target, skip_special_tokens=True)[0]

            save_data['model_inputs'].append(description)
            save_data['preds'].append(prediction)
            save_data['gts'].append(ground_truth)

            if step == 20:
                break

        prec, rec, acc, f1 = compute_metrics(save_data['preds'], save_data['gts'], 'entity_matching')

        logger.info(
            f"Epoch {epoch}, Step {step}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        with open(f'outputs/metrics/metrics_entity_matching_epoch{epoch}.json', 'w') as fp:
            json.dump(trial_metrics, fp)
            logger.info(f"Saved to metrics_entity_matching_epoch{epoch}.json")

        with open(f'outputs/data/data_entity_matching_epoch{epoch}.json', 'w') as fp:
            json.dump(save_data, fp)
            logger.info(f"Saved to data_entity_matching_epoch{epoch}.json")