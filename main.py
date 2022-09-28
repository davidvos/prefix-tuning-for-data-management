import json
import yaml

import logging
import numpy as np
import torch
import wandb
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from dataset import PrefixDataset
from prefix_model import PrefixTuning
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from utils.utils import compute_metrics, setup_logger, parse_args

# Initialize a logger and parse given arguments
logger = logging.getLogger(__name__)
args = parse_args()

# Obtain the dataset name and directory from the specified path
dataset_name = args.data_dir.split('/')[-1]
data_dir = str(Path(args.data_dir).resolve())

with open('config.yaml') as f:
    wandb_config = yaml.safe_load(f)

# Initialize WandB
# Create an account at https://wandb.ai/, create a project and specify your details in 'config.yaml'
# A quickstart guide is available at https://docs.wandb.ai/quickstart 
# wandb.init(project=wandb_config['wandb']['project_name'], entity=wandb_config['wandb']['entity'],
#     config={'lr': args.lr, 'batch_size': args.batch_size, 'prefix_length': args.prefix_size, 'epochs': args.n_epochs, 'dataset': dataset_name, 'n_samples': args.n_samples, 'finetune_type': args.finetune_type})

setup_logger(args.output_dir)
logger.info(json.dumps(vars(args), indent=4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():
    # Pre-Trained T5 Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    tokenizer.max_length = 200

    # Pre-Trained T5 Model
    model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

    # Resize PLM's Embedding Layer
    model.resize_token_embeddings(len(tokenizer))

    # Freeze all LLM parameters and initialize prefix-tuning model
    if args.finetune_type == 'prefix':
        # Freeze LM
        for param in model.parameters():
            param.requires_grad=False

        prefix_model = PrefixTuning(model.config, args.prefix_size).to(device)

    # Initialize datasets and dataloaders
    dataset_train = PrefixDataset(tokenizer, data_dir, args.prefix_size, 'train', args.finetune_type, args.n_samples)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)

    dataset_eval = PrefixDataset(tokenizer, data_dir, args.prefix_size, 'validation', args.finetune_type, args.n_samples)
    dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, collate_fn=dataset_eval.collate_fn)

    dataset_test = PrefixDataset(tokenizer, data_dir, args.prefix_size, 'test', args.finetune_type)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=dataset_test.collate_fn)

    # Initialize optimizer and learning rate scheduler
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
    best_acc_score = 0

    for epoch in range(args.n_epochs):

        trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

        if args.finetune_type == 'prefix':
            prefix_model.train()
        elif args.finetune_type == 'fine':
            model.train()

        for step, (description, attention_mask, target) in enumerate(dataloader_train):

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
        
        # Evaluate on entire validation set after an epoch of training
        with torch.no_grad():

            save_data = {}
            save_data['model_inputs'] = []
            save_data['preds'] = []
            save_data['gts'] = []

            if args.finetune_type == 'prefix':
                prefix_model.eval()
            else:
                model.eval()

            wandb.log({'status': f'evaluating epoch {epoch}'})

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
                ground_truth = tokenizer.batch_decode(target, skip_special_tokens=True)[0]

                # Store input, output and target data
                save_data['model_inputs'].append(description)
                save_data['preds'].append(output_text)
                save_data['gts'].append(ground_truth)
            
            # Compuate validation metrics
            prec, rec, acc, f1 = compute_metrics(save_data['preds'], save_data['gts'], 'error_detection')
            
            wandb.log({'eval_precision': prec, 'eval_recall': rec, 'eval_accuracy': acc, 'eval_f1': f1})

            logger.info(
                f"Epoch {epoch}, Step {step}\n"
                f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
            )
            trial_metrics["rec"].append(rec)
            trial_metrics["prec"].append(prec)
            trial_metrics["acc"].append(acc)
            trial_metrics["f1"].append(f1)

            # Write saved data and matrics to json files
            with open(f'outputs/metrics/metrics_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_eval.json', 'w') as fp:
                json.dump(trial_metrics, fp)
            logger.info(f"Saved to metrics_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_eval.json")

            with open(f'outputs/data/data_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_eval.json', 'w') as fp:
                json.dump(save_data, fp)
            logger.info(f'outputs/data/data_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_eval.json')

            # Store best model based on validation f1 or accuracy, based on the current task
            if (dataset_name == 'Restaurant' or dataset_name == 'Buy') and acc >= best_acc_score:
                best_acc_score = acc
                wandb.log({'highest_eval_acc': best_acc_score})
                if args.finetune_type == 'prefix':
                    torch.save(prefix_model.state_dict(), f'outputs/models/entity_matching_prefix_dataset{dataset_name}_lr{args.lr}_best.pt')
                elif args.finetune_type == 'fine':
                    torch.save(model.state_dict(), f'outputs/models/entity_matching_fine_dataset{dataset_name}_lr{args.lr}_best.pt')
            
            if (dataset_name != 'Restaurant' and dataset_name != 'Buy') and f1 >= best_f1_score:
                best_f1_score = f1
                wandb.log({'highest_eval_f1': best_f1_score})
                if args.finetune_type == 'prefix':
                    torch.save(prefix_model.state_dict(), f'outputs/models/entity_matching_prefix_dataset{dataset_name}_lr{args.lr}_best.pt')
                elif args.finetune_type == 'fine':
                    torch.save(model.state_dict(), f'outputs/models/entity_matching_fine_dataset{dataset_name}_lr{args.lr}_best.pt')

            wandb.log({'status': f'saved results epoch {epoch}'})

    # After training, run inference on the test set using the best validation model 
    trial_metrics = {"prec": [], "rec": [], "f1": [], "acc": []}

    save_data = {}
    save_data['model_inputs'] = []
    save_data['preds'] = []
    save_data['gts'] = []

    # Load best model 
    if args.finetune_type == 'prefix':
        prefix_model.load_state_dict(torch.load(f'outputs/models/entity_matching_prefix_dataset{dataset_name}_lr{args.lr}_best.pt'))
        prefix_model.eval()
    elif args.finetune_type == 'fine':
        model.load_state_dict(torch.load(f'outputs/models/entity_matching_fine_dataset{dataset_name}_lr{args.lr}_best.pt'))
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

            ground_truth = tokenizer.batch_decode(target, skip_special_tokens=True)[0]

            save_data['model_inputs'].append(description)
            save_data['preds'].append(output_text)
            save_data['gts'].append(ground_truth)

        # Compuate validation metrics
        prec, rec, acc, f1 = compute_metrics(save_data['preds'], save_data['gts'], 'error_detection')

        wandb.log({'test_precision': prec, 'test_recall': rec, 'test_accuracy': acc, 'test_f1': f1})

        logger.info(
            f"Test set results\n"
            f"Prec: {prec:.3f} Recall: {rec:.3f} Acc: {acc:.3f} F1: {f1:.3f}"
        )
        trial_metrics["rec"].append(rec)
        trial_metrics["prec"].append(prec)
        trial_metrics["acc"].append(acc)
        trial_metrics["f1"].append(f1)

        # Write saved data and matrics to json files
        with open(f'outputs/metrics/metrics_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_test.json', 'w') as fp:
            json.dump(trial_metrics, fp)
            logger.info(f"Saved to metrics_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_test.json")

        with open(f'outputs/data/data_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_test.json', 'w') as fp:
            json.dump(save_data, fp)
            logger.info(f"Saved to data_entity_matching_{args.finetune_type}_epoch{epoch}_dataset{dataset_name}_lr{args.lr}_test.json")
        
        if args.finetune_type == 'prefix':
            prefix_model.zero_grad()
        else:
            model.zero_grad()

if __name__ == '__main__':
    main()