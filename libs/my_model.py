"""
Υβριδικό μοντέλο CNN + Transformer.
Το ResNet50 εξάγει χωρικά χαρακτηριστικά ανά frame,
και ένας Transformer Encoder μοντελοποιεί τη χρονική σχέση μεταξύ των frames.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CNN_Transformer(nn.Module):

    """Υβριδική αρχιτεκτονική: ResNet50 ανά frame -> Transformer Encoder στον χρόνο -> ταξινόμηση σε 7 συναισθήματα."""

    def __init__(self, num_classes=7, num_frames=16):

        """Στήνει τα τρία μέρη του μοντέλου: το ResNet50 backbone (χωρίς το FC), τον Transformer Encoder με positional embedding, και τον τελικό classifier."""

        super(CNN_Transformer, self).__init__()
        
        print("Initializing Custom Hybrid Model: ResNet50 + Transformer Encoder")
        
        # 1. BACKBONE: ResNet50 (Pretrained on ImageNet)
        # Χρησιμοποιούμε το ResNet50 που είναι πιο βαθύ από το ResNet18
        resnet = models.resnet50(weights="DEFAULT")
        
        # Αφαιρούμε το τελευταίο FC layer και το Pooling για να πάρουμε τα features
        modules = list(resnet.children())[:-1] 
        self.backbone = nn.Sequential(*modules)
        
        self.feature_dim = 2048 # Το ResNet50 βγάζει 2048 κανάλια

        # 2. TEMPORAL: Transformer Encoder
        # Ορίζουμε έναν Transformer που θα βλέπει τη σειρά των frames
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Positional Embedding: Για να ξέρει το μοντέλο ποιο frame είναι 1ο, 2ο, κλπ.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, self.feature_dim))

        # 3. CLASSIFIER: Η τελική απόφαση
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        """Περνά κάθε frame από το ResNet50, προσθέτει χρονική πληροφορία μέσω Transformer, παίρνει τον μέσο όρο στον χρόνο των frames και ταξινομεί. Είσοδος [B, 3, T, 224, 224] -> έξοδος [B, 7]."""

        # x shape: [Batch, 3, Frames, 224, 224]
        b, c, t, h, w = x.size()

        # Αναδιάταξη για να περάσει από το 2D CNN
        # Ενώνουμε το Batch και το Time -> [Batch * Frames, 3, 224, 224]
        x = x.transpose(1, 2).contiguous().view(b * t, c, h, w)

        # Πέρασμα από το ResNet50
        features = self.backbone(x) # Output: [Batch*Frames, 2048, 1, 1]
        features = features.view(b, t, -1) # Επαναφορά σε [Batch, Frames, 2048]

        # Προσθήκη Positional Embedding (για να καταλάβει τη χρονική σειρά)
        features = features + self.pos_embedding[:, :t, :]

        # Πέρασμα από τον Transformer
        trans_features = self.transformer(features)

        # Global Average Pooling στον χρόνο (παίρνουμε τον μέσο όρο των frames)
        out = trans_features.mean(dim=1) 
        
        # Τελική ταξινόμηση
        out = self.dropout(out)
        out = self.fc(out)

        return out