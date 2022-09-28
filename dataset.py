from torch.utils.data import Dataset
import torch
import random

from utils import data_utils, constants

class PrefixDataset(Dataset):
    
    def __init__(self, tokenizer, data_dir, prefix_size, split='train', finetune_type='prefix', n_samples=0):
        
        self.prefix_size = prefix_size

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
        
        if n_samples > 0:
            descriptions = descriptions[:n_samples]
            targets = targets[:n_samples]

        self.descriptions = descriptions
        self.targets = targets

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.descriptions[idx]), self.tokenizer.encode(self.targets[idx])

    def collate_fn(self, batch):
        # """
        # Same Sequence Length on Same Batch
        # """
        max_len_data=0
        max_len_label=0
        for description, target in batch:
            if len(description)>max_len_data: max_len_data=len(description)
            if len(target)>max_len_label: max_len_label=len(target)
                
        attn_masks=[]
        targets=[]
        descriptions=[]
        for description, target in batch:

            description.extend([self.tokenizer.pad_token_id]*(max_len_data-len(description)))
            descriptions.append(description)
            
            attn_mask=[int(e!=self.tokenizer.pad_token_id) for e in description]
            attn_masks.append(attn_mask)
            
            target.extend([-100]*(max_len_label-len(target)))
            targets.append(target)

        return torch.LongTensor(descriptions), torch.LongTensor(attn_masks), torch.LongTensor(targets)