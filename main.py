import argparse
import torch
from pathlib import Path
from dataset import PrefixDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import logging
import json
import wandb

from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler

from utils.utils import compute_metrics, setup_logger, parse_args
from prefix_model import PrefixTuning

logger = logging.getLogger(__name__)

args = parse_args()

dataset_name = args.data_dir.split('/')[-1]
# wandb.init(project="prefix-tuning-entity-matching", entity="dvos", config={'lr': args.lr, 'batch_size': args.batch_size, 'prefix_length': args.prefix_size, 'epochs': args.n_epochs, 'dataset': dataset_name, 'finetune_type': args.finetune_type})

setup_logger(args.output_dir)
logger.info(json.dumps(vars(args), indent=4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Pre-Trained T5 Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.max_length = 200

# Pre-Trained T5 Model
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# Resize PLM's Embedding Layer
model.resize_token_embeddings(len(tokenizer))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'T5 parameters: {pytorch_total_params}')

if args.finetune_type == 'prefix':
    # Freeze LM
    for param in model.parameters():
        param.requires_grad=False

    prefix_model = PrefixTuning(model.config, args.prefix_size) 
    prefix_model = prefix_model.to(device)
    prefix_total_params = sum(p.numel() for p in prefix_model.parameters())
    print(f'Prefix parameters: {prefix_total_params}')

data_dir = data_dir = str(Path(args.data_dir).resolve())

dataset_train = PrefixDataset(tokenizer, data_dir, args.prefix_size, 'train', args.finetune_type)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)

dataset_eval = PrefixDataset(tokenizer, data_dir, args.prefix_size, 'validation', args.finetune_type)
dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, collate_fn=dataset_eval.collate_fn)

dataset_test = PrefixDataset(tokenizer, data_dir, args.prefix_size, 'test', args.finetune_type)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=dataset_eval.collate_fn)

if args.finetune_type == 'prefix':
    optimizer = AdamW(prefix_model.parameters(), lr=args.lr)
elif args.finetune_type == 'fine':
    optimizer = AdamW(model.parameters(), lr=args.lr)

if args.finetune_type == 'prefix' or args.finetune_type == 'fine':
    num_training_steps = args.n_epochs * len(dataloader_train)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

best_f1_score = 0

for epoch in range(args.n_epochs):

    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

    if args.finetune_type == 'prefix':
        prefix_model.train()
    elif args.finetune_type == 'fine':
        model.train()

    for step, (description, attention_mask, target) in enumerate(dataloader_train):

        if args.finetune_type == 'prompt':
            break

        description = description.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        if args.finetune_type == 'prefix':
            prefix = prefix_model(batch_size=description.shape[0], device=device)
            outputs = model(input_ids=description, attention_mask=attention_mask, labels=target, prompt=prefix)
        else:
            outputs = model(input_ids=description, attention_mask=attention_mask, labels=target)
        
        loss = outputs.loss
        wandb.log({'training_loss': loss})
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if args.finetune_type == 'prefix':
            prefix_model.zero_grad()
        else:
            model.zero_grad()

    with torch.no_grad():

        save_data = {}
        save_data['model_inputs'] = []
        save_data['preds'] = []
        save_data['gts'] = []

        if args.finetune_type == 'prefix':
            prefix_model.eval()
        else:
            model.eval()

        for step, (description, attention_mask, target) in enumerate(dataloader_eval):

            description = description.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            if args.finetune_type == 'prefix':
                prefix = prefix_model(batch_size=description.shape[0], device=device)
                outputs = model.generate(description, max_length=100, num_beams=5, early_stopping=True, prompt=prefix)
            else:
                outputs = model.generate(description, max_length=100, num_beams=5, early_stopping=True)
            
            description = tokenizer.batch_decode(description, skip_special_tokens=True)[0]
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            if args.finetune_type == 'prompt':
                if output_text == 'True':
                    output_text = 'Yes'
                else:
                    output_text = 'No'

            ground_truth = tokenizer.batch_decode(target, skip_special_tokens=True)[0]

            save_data['model_inputs'].append(description)
            save_data['preds'].append(output_text)
            save_data['gts'].append(ground_truth)

        prec, rec, acc, f1 = compute_metrics(save_data['preds'], save_data['gts'], 'entity_matching')

        wandb.log({'eval_precision': prec, 'eval_recall': rec, 'eval_accuracy': acc, 'eval_f1': f1})

        logger.info(
            f"Epoch {epoch}, Step {step}\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        with open(f'outputs/metrics/metrics_entity_matching_prefix_epoch{epoch}_dataset{dataset_name}_eval.json', 'w') as fp:
            json.dump(trial_metrics, fp)
            logger.info(f"Saved to metrics_entity_matching_prefix_epoch{epoch}_dataset{dataset_name}_eval.json")

        with open(f'outputs/data/data_entity_matching_prefix_epoch{epoch}_dataset{dataset_name}_eval.json', 'w') as fp:
            json.dump(save_data, fp)
            logger.info(f'outputs/data/data_entity_matching_prefix_epoch{epoch}_dataset{dataset_name}_eval.json')

        if f1 >= best_f1_score:
            if args.finetune_type == 'prefix':
                torch.save(prefix_model.state_dict(), f'outputs/models/entity_matching_prefix_dataset{dataset_name}_best.pt')
            elif args.finetune_type == 'fine':
                torch.save(model.state_dict(), f'outputs/models/entity_matching_fine_dataset{dataset_name}_best.pt')
                
    if args.finetune_type == 'prompt':
        break

trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

save_data = {}
save_data['model_inputs'] = []
save_data['preds'] = []
save_data['gts'] = []

if args.finetune_type == 'prefix':
    prefix_model.load_state_dict(torch.load(f'outputs/models/entity_matching_prefix_dataset{dataset_name}_best.pt'))
    prefix_model.eval()
elif args.finetune_type == 'fine':
    model.load_state_dict(torch.load(f'outputs/models/entity_matching_fine_dataset{dataset_name}_best.pt'))
    model.eval()

with torch.no_grad():

    for step, (description, attention_mask, target) in enumerate(dataloader_test):

        description = description.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        if args.finetune_type == 'prefix':
            prefix = prefix_model(batch_size=description.shape[0], device=device)
            outputs = model.generate(description, max_length=100, num_beams=5, early_stopping=True, prompt=prefix)
        else:
            outputs = model.generate(description, max_length=100, num_beams=5, early_stopping=True)
        
        description = tokenizer.batch_decode(description, skip_special_tokens=True)[0]
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if args.finetune_type == 'prompt':
            if output_text == 'True':
                output_text = 'Yes'
            else:
                output_text = 'No'

        ground_truth = tokenizer.batch_decode(target, skip_special_tokens=True)[0]

        save_data['model_inputs'].append(description)
        save_data['preds'].append(output_text)
        save_data['gts'].append(ground_truth)

    prec, rec, acc, f1 = compute_metrics(save_data['preds'], save_data['gts'], 'data_imputation')

    wandb.log({'test_precision': prec, 'test_recall': rec, 'test_accuracy': acc, 'test_f1': f1})

    logger.info(
        f"Test set results\n"
        f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
    )
    trial_metrics["rec"].append(rec)
    trial_metrics["prec"].append(prec)
    trial_metrics["acc"].append(acc)
    trial_metrics["f1"].append(f1)

    with open(f'outputs/metrics/metrics_entity_matching_epoch{epoch}_dataset{dataset_name}_test.json', 'w') as fp:
        json.dump(trial_metrics, fp)
        logger.info(f"Saved to metrics_entity_matching_epoch{epoch}_dataset{dataset_name}_test.json")

    with open(f'outputs/data/data_entity_matching_epoch{epoch}_dataset{dataset_name}_test.json', 'w') as fp:
        json.dump(save_data, fp)
        logger.info(f"Saved to data_entity_matching_epoch{epoch}_dataset{dataset_name}_test.json")
    
    if args.finetune_type == 'prefix':
            prefix_model.zero_grad()
    else:
        model.zero_grad()
