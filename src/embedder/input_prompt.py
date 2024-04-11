import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange

class SoftPrompt(nn.Module):
    def __init__(self, 
                in_dim: int = 1024,
                embed_dim: int = 1024,
                dropout: float = 0.0,
                num_hidden_layers: int = 2):
        """mapping the inputs 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftPrompt, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        layer_stack=[]
        for _ in range(self.num_hidden_layers-1):
            layer_stack.extend(
                [
                    torch.nn.Linear(
                        in_features=self.in_dim,
                        out_features=self.embed_dim
                    ),
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.ReLU(),
                    #torch.nn.Dropout(p=self.dropout)
                ]
            )
        layer_stack.extend(
            [
                torch.nn.Linear(
                    in_features=self.embed_dim if self.num_hidden_layers>1 else self.in_dim,
                    out_features=self.embed_dim
                ),
                torch.nn.LayerNorm(self.embed_dim),
                torch.nn.ReLU(),
                #torch.nn.Dropout(p=self.dropout)
            ]
        )
        self.soft_prompt = torch.nn.Sequential(*layer_stack)
    
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
            
    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> torch.tensor:

        inputs = batch['inputs']

        inputs_stacked = self._stack_inputs(tensor=inputs)
        
        return self._unstack_inputs(
            tensor=self.soft_prompt(inputs_stacked),
            b=inputs.size()[0]
        ), None, None