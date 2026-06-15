"""
Wrappers για τα προεκπαιδευμένα μοντέλα: TimeSformer και VideoMAE (από το
HuggingFace, pretrained σε Kinetics-400) και το επίσημο Former-DFER. Κάθε wrapper προσαρμόζει
το backbone ώστε να βγάζει 7 κλάσεις (συναισθήματα DFEW) και να δέχεται τη μορφή tensor του dataset.

"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from libs import ST_Former

class TimeSformerDFEW(nn.Module):

    """TimeSformer (base, pretrained σε Kinetics-400) με νέο classifier για τα 7 συναισθήματα του DFEW."""

    def __init__(self, num_classes=7, pretrained=True):
        super(TimeSformerDFEW, self).__init__()
        
        print(f"Loading TimeSformer (Pretrained: {pretrained})...")
        
        # Χρησιμοποιούμε το TimeSformer-base
        model_name = "facebook/timesformer-base-finetuned-k400"
        
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        # Το TimeSformer βγάζει ένα vector διάστασης 768.
        # Πρέπει να το μετατραπεί στα 7 συναισθήματα.
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # Το TimeSformer του HuggingFace περιμένει είσοδο [Batch, Frames, Channels, Height, Width]
        # Το DFEW dataset δίνει [Batch, Channels, Frames, Height, Width]
        # Πρέπει να αλλαχθεί η σειρά των διαστάσεων (Permute)
        
        # Από (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Πέρασμα από το μοντέλο
        outputs = self.backbone(pixel_values=x)
        
        # Παίρνει το feature vector (last_hidden_state)
        
        cls_token = outputs.last_hidden_state[:, 0]
        
        # Ταξινόμηση
        logits = self.classifier(cls_token)
        
        return logits
    


class VideoMAEDFEW(nn.Module):

    """VideoMAE (base, pretrained σε Kinetics-400) με νέο classifier για τα 7 συναισθήματα του DFEW."""

    def __init__(self, num_classes=7, pretrained=True):

        """Φορτώνει το VideoMAE backbone και προσθέτει ένα γραμμικό classifier 768 -> 7."""

        super(VideoMAEDFEW, self).__init__()
        
        print(f"Loading VideoMAE (Pretrained: {pretrained})...")
        
        # Χρησιμοποιείται το VideoMAE base μοντέλο
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)

        # Το VideoMAE base βγάζει output size 768
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # Το VideoMAE θέλει: [Batch, Frames, Channels, Height, Width]
        # Το Dataset δίνει:  [Batch, Channels, Frames, Height, Width]
        
        # Permute για να έρθουν στη σωστή σειρά
        x = x.permute(0, 2, 1, 3, 4)
        
        # Πέρασμα από το μοντέλο
        outputs = self.backbone(pixel_values=x)
        
        
        # Γίνεται  Mean Pooling (Μέσος όρος) σε όλα τα tokens
        last_hidden_state = outputs.last_hidden_state
        mean_pooling = last_hidden_state.mean(dim=1) 
        
        # Ταξινόμηση
        logits = self.classifier(mean_pooling)
        
        return logits
    
class FormerDFER_Wrapper(nn.Module):

    """Wrapper για το επίσημο μοντέλο Former-DFER, ώστε να καλείται με το ίδιο interface όπως τα υπόλοιπα."""

    def __init__(self, num_classes=7):
        super().__init__()
        print("Loading Official Former-DFER Model from libs...")
        # Δεν περνάμε ορίσματα, το αρχικοποιεί όπως το GitHub
        self.model = ST_Former.GenerateModel()

    def forward(self, x):
        return self.model(x)