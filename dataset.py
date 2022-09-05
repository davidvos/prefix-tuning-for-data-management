from torch.utils.data import Dataset
import torch

from utils import data_utils, constants

class PrefixDataset(Dataset):
    
    def __init__(self, tokenizer, data_dir, split='train'):
        
        self.tokenizer = tokenizer

        # Read pandas DF datasets
        pd_data_files = data_utils.read_data(
            data_dir = data_dir,
            add_prefix = False,
            max_train_samples = -1,
            sep_tok='.',
            nan_tok='nan',
        )

        examples = pd_data_files[split]

        descriptions = examples['text'].apply(lambda x: x.strip())
        targets = examples['label_str'].apply(lambda x: x.rstrip('\n'))

        self.descriptions = descriptions + ' answer: '
        self.final_examples = descriptions + ' answer: ' + targets
        self.targets = targets

    def __len__(self):
        return len(self.final_examples)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.descriptions[idx]), self.tokenizer.encode(self.final_examples[idx]), self.tokenizer.encode(self.targets[idx])

    def collate_fn(self, batch):
        # """
        # Same Sequence Length on Same Batch
        # """
        max_len_data=0
        max_len_label=0
        for description, sample, target in batch:
            if len(sample)>max_len_data: max_len_data=len(sample)
            if len(target)>max_len_label: max_len_label=len(target)
                
        samples=[]
        # attn_masks=[]
        targets=[]
        descriptions=[]
        for description, sample, target in batch:

            description.extend([self.tokenizer.eos_token_id]*(max_len_data-len(description)))
            descriptions.append(description)

            sample.extend([self.tokenizer.eos_token_id]*(max_len_data-len(sample)))
            samples.append(sample)
            
            # attn_mask=[int(e!=tokenizer.pad_token_id) for e in data]
            # attn_masks.append(attn_mask)
            
            target.extend([-100]*(max_len_label-len(target)))
            targets.append(target)

        return torch.LongTensor(descriptions), torch.LongTensor(samples), torch.LongTensor(targets)