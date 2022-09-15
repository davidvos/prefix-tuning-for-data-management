import torch
import torch.nn as nn

class PrefixTuning(nn.Module):
    """
    """
    def __init__(self, pretrained_config, prompt_len=100, hidden_dim=512):
        super().__init__()
        
        # Config of Pre-Trained LM
        self.pretrained_config=pretrained_config
        
        # torch.tensor([0, 1, 2, .. , prompt_len-1])
        self.pre_prompt=torch.arange(prompt_len)
        # Embedding
        self.embd=nn.Embedding(num_embeddings=prompt_len, embedding_dim=pretrained_config.d_model)
        # Reparameterization
        self.reparam=nn.Sequential(
            nn.Linear(pretrained_config.d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, pretrained_config.d_model)
        )
        
    def forward(self, batch_size, device):
        # Shape: batch_size, prompt_len
        prompt=self.pre_prompt.unsqueeze(0).expand(batch_size, -1).to(device)
        # Shape: batch_size, prompt_len, d_model
        prompt=self.embd(prompt)
        # Shape: batch_size, prompt_len, d_model
        prompt=self.reparam(prompt)
        
        return prompt