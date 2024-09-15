import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from dt.transformer import TransformerDecoder

to_np = lambda x: x.detach().cpu().numpy()

class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.drop)

        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(3136, config.embed_dim), nn.Tanh())
        
        self.action_emb = nn.Sequential(nn.Embedding(config.n_action, config.embed_dim), nn.Tanh())
        
        self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_len*2, config.embed_dim))
        self.decoder = TransformerDecoder(config)

        self.ln = nn.LayerNorm(config.embed_dim)
        self.decoder_head = nn.Linear(config.embed_dim, config.n_action, bias=False)

        self.seq_len = config.seq_len
        self.embed_dim = config.embed_dim

        self.device = config.device
    
    def forward(self, states, actions):
        # states (B, T, C, H, W)
        # actions (B, T)
        B, ST, _, _, _ = states.shape
        _, AT = actions.shape
        states_embed = self.state_encoder(states.reshape(-1, 4, 84, 84).contiguous())
        states_embed = states_embed.reshape(B, ST, self.embed_dim)

        if AT != 0:
            action_embed = self.action_emb(actions)

            tok = torch.zeros((B, ST+AT, self.embed_dim), dtype=torch.float32, device=self.device)
            tok[:,::2,:] = states_embed
            tok[:,1::2,:] = action_embed
        elif AT == 0:
            tok = states_embed

        pos = self.pos_emb[:,:tok.shape[1],:]

        x = self.decoder(self.dropout(tok + pos))

        if AT != 0:
            x = x[:,::2,:]

        logits = self.decoder_head(self.ln(x))
        return logits
    
    def infer(self, s, a, batch=False):
        assert isinstance(s, np.ndarray)
        assert isinstance(a, np.ndarray)
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32, device=self.device) / 255.0
            a = torch.tensor(a, dtype=torch.long, device=self.device)
            if not batch:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            p = self.forward(s, a)[:,-1] # last action
            if not batch:
                p = p.squeeze(0)
            p = to_np(p)
        return p

class DecisionTransformerConfig:
    def __init__(self):
        self.embed_dim = 512
        self.n_layer = 4
        self.n_head = 4
        self.drop = 0.1
        self.n_action = None
        self.seq_len = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
if __name__ == "__main__":
    config = DecisionTransformerConfig()
    config.n_action = 4
    config.device = 'cpu'

    dt = DecisionTransformer(config)

    batch = 64
    seq_len = config.seq_len
    n_action = config.n_action

    states = torch.rand((batch, seq_len, 4, 84, 84), dtype=torch.float32)
    actions = torch.randint(0, n_action, (batch, seq_len-1))
    y = dt(states, actions)
    print(y.shape)

    states = np.random.randint(0, 255, (1, 4, 84, 84), dtype=np.uint8)
    actions = np.array([], dtype=int)
    y = dt.infer(states, actions)
    print(y)

    states = np.random.randint(0, 255, (32, 4, 84, 84), dtype=np.uint8)
    actions = np.random.randint(0, n_action, (31,), dtype=int)
    y = dt.infer(states, actions)
    print(y)
