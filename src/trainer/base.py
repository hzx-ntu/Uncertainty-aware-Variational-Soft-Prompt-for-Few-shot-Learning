#!/usr/bin/env python3

from typing import Tuple, Optional, List, Dict
import torch
import transformers


class Trainer(transformers.Trainer):
    def __init__(
        self,
        is_deepspeed: bool = False,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = "Trainer"
        self.is_deepspeed = is_deepspeed
        self.is_promptembed = False #hzx added
        self.is_promptembed_vae = False #hzx added
        self.prompt_embeds = [] #hzx added
        self.prompt_vars = [] #hzx added
        self.is_masked = False #hzx added
        self.masked_ratio = [] #hzx added

    def prediction_step(
        self,
        model,
        batch,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        batch = self._move_batch_to_device(batch=batch)
        
        with torch.no_grad():
            (loss, outputs) = self.compute_loss(
                model=model,
                batch=batch,
                return_outputs=True
            )
        #hzx added
        if self.is_promptembed:
           self.prompt_embeds.append(outputs['prompt_embed_out'])
        
        #hzx added
        if self.is_promptembed_vae:
           self.prompt_vars.append(outputs['prompt_var'])
        
        #hzx added
        if self.is_masked:
           self.masked_ratio.append(outputs['mask_ratios'])

        if not prediction_loss_only and 'labels' in batch:
            return (loss, outputs['decoding_logits'], batch['labels'])
        
        else:
            return (loss, None, None)

    def compute_loss(
        self,
        model,
        batch,
        return_outputs=False,
        **kwargs
        ):
        batch = self._move_batch_to_device(batch=batch)

        if isinstance(
            model,
            (
                torch.nn.DataParallel, 
                torch.nn.parallel.DistributedDataParallel
            )
        ) or self.is_deepspeed:
            (losses, outputs) = model.module.compute_loss(
                batch=batch,
                return_outputs=True
            )
        
        else:
            (losses, outputs) = model.compute_loss(
                batch=batch,
                return_outputs=True
            )
        
        loss = losses['loss'] if 'loss' in losses.keys() else sum(losses.values())
        
        return (loss, outputs) if return_outputs else loss

    def _move_batch_to_device(
        self,
        batch
        ) -> Dict[str, torch.tensor]:
        batch = self._prepare_inputs(batch)
        
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.long).to(batch["inputs"].device)
        
        return self._prepare_inputs(batch)