from rdkit import Chem
import os
import pickle
import numpy as np

import torch
from einops import rearrange

from ctsynther.utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph
from ctsynther.utils.smiles_utils import canonical_smiles, canonical_smiles_with_am, remove_am_without_canonical, \
    extract_relative_mapping, get_nonreactive_mask, randomize_smiles_with_am


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, feature, label):
        
        feature = rearrange(feature, 'l b -> b l')
        feature = feature / torch.norm(feature, dim=-1, keepdim=True) 
        similarity = torch.mm(feature, feature.T) 
        
        
        loss_contrastive = torch.sum((1-label) * torch.pow(similarity, 2) +     
                                       (label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))
       
        return loss_contrastive

