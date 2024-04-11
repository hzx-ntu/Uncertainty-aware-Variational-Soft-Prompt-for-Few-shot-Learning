import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange

class CustomLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0,
                 bidirectional=False):
        super(CustomLSTM, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

        
        self.activation = nn.ReLU()

    def forward(self, input, hx=None):
        self.batch_size = input.size()[0]
        if hx is None:
            h_0 = input.new_zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size,
                                   requires_grad=False)
            c_0 = input.new_zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size,
                                   requires_grad=False)
        else:
            h_0, c_0 = hx

        
        self.hidden = (h_0, c_0)
        lstm_output, self.hidden = self.lstm(input, self.hidden)
        lstm_output = self.activation(lstm_output)

        return lstm_output, self.hidden

class SoftPromptLSTM(nn.Module):
    def __init__(self, 
                in_dim: int = 1024,
                embed_dim: int = 1024,
                dropout: float = 0.0,
                num_hidden_layers: int = 1):
        """mapping the inputs 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftPromptLSTM, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.soft_prompt = torch.nn.LSTM(input_size=in_dim, hidden_size=embed_dim, num_layers=num_hidden_layers, batch_first=True)
        #self.soft_prompt = CustomLSTM(input_size=in_dim, hidden_size=embed_dim, num_layers=num_hidden_layers, batch_first=True)
        self.fc_mean = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim) # mean
        self.fc_var = torch.nn.Linear(in_features=embed_dim, out_features=embed_dim) # var
        self.is_test = False
    
    def _stack_inputs(
        self,
        tensor
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='b s e -> (b s) e'
        )

    def _unstack_inputs(
        self,
        tensor,
        b
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='(b s) e -> b s e',
            b=b
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        if self.is_test:
           return mu
        else:
           return mu + eps * std        
            
    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> torch.tensor:

        inputs = batch['inputs']
        
        h0 = torch.zeros((self.num_hidden_layers, inputs.size(0), self.embed_dim), device=torch.device('cuda'))
        c0 = torch.zeros((self.num_hidden_layers, inputs.size(0), self.embed_dim), device=torch.device('cuda'))
        
        output, (hn, cn) = self.soft_prompt(inputs, (h0, c0))
        
        '''
        mu = self.fc_mean(hn.squeeze())
        log_var = self.fc_var(hn.squeeze())
        
        z = self.reparameterize(mu, log_var)
        
        return self._unstack_inputs(tensor=z,b=inputs.size()[0]),log_var, mu
        '''
        return hn.permute(1,0,2), None, None