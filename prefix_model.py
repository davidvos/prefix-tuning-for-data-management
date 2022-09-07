import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    """
    """
    def __init__(self, base_config, prefix_size, hidden_dim=512):
        super().__init__()

        # Config of Base (Pre-Trained) LM
        self.base_config=base_config

        # Input: 0, 1, 2 ... preseqlen
        self.preseq=torch.arange(prefix_size)
        # Embedding
        self.embd=nn.Embedding(prefix_size,base_config.n_embd)
        # Reparam
        self.reparam=nn.Sequential(
            nn.Linear(base_config.n_embd,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,2*base_config.n_layer*base_config.n_embd)
        )

    def forward(self, batch_size, device):
        # batch_size, preseqlen
        preseq=self.preseq.unsqueeze(0).expand(batch_size,-1).to(device)
        # batch_size, preseqlen, n_embd
        preseq=self.embd(preseq)
        # batch_size, preseqlen, 2*n_layer*n_embd
        preseq=self.reparam(preseq)
        # batch_size, preseqlen, 2*n_layer, n_head, n_embd/n_head
        preseq=preseq.reshape(batch_size,len(self.preseq),2*self.base_config.n_layer,self.base_config.n_head,int(self.base_config.n_embd/self.base_config.n_head))
        # 2*n_layer, batch_size, n_head, preseqlen, n_embd/n_head
        past_key_values=preseq.permute(2,0,3,1,4)

        return past_key_values.split(2)