import torch
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

def multi_dim_softmax(logits: torch.Tensor, tau_add_one: torch.Tensor,   eps: float = 1e-10 ) -> torch.Tensor:
    # logits (M,N)
    # tau_add_one   (M,)
    # return (M,N)
    M,N= logits.size(0),logits.size(1)
    tau = tau_add_one + torch.ones_like(tau_add_one).to(tau_add_one.device) # (M,)
    gumbels = - torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()#(M,N), ~Gumbel(0,1)
    gumbels  = gumbels * tau_add_one.unsqueeze(1).expand(-1,N) #(M,N)

    logits_t = torch.div(logits + gumbels, tau.unsqueeze(1).expand(-1,N)) # (M,N)

    sft = torch.nn.functional.softmax(logits_t,dim=1)  # (M,N)
    return sft  # (M,N)





