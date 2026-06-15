"""
Επίσημο μοντέλο Former-DFER (κώδικας των δημιουργών). Συνδυάζει τον S-Former (χωρικός
encoder) με τον T-Former (χρονικός encoder) και έναν τελικό classifier για 7 συναισθήματα.

"""

import torch
from torch import nn
from libs.S_Former import spatial_transformer
from libs.T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, 7)

    def forward(self, x):

        x = self.s_former(x)
        x = self.t_former(x)
        x = self.fc(x)
        return x

