"""
Script για την μέτρηση των παραμέτρων όλων των μοντέλων.

ΕΚΤΕΛΕΣΗ: python count_params.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


from libs import my_model          # Υβριδικό (CNN_Transformer)
from Loading_models import my_models         # TimeSformer, VideoMAE, Former-DFER


def count_parameters(model):
    """Μετράει μόνο τις εκπαιδεύσιμες παραμέτρους (trainable)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """Μετράει ΟΛΕΣ τις παραμέτρους (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def report(name, model):
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"{name:<20} | Σύνολο: {total:>14,} | Εκπαιδεύσιμες: {trainable:>14,} | {total/1e6:.2f}M")


if __name__ == "__main__":
    NUM_CLASSES = 7
    NUM_FRAMES = 16

    print("=" * 80)
    print(f"{'ΜΟΝΤΕΛΟ':<20} | {'ΣΥΝΟΛΟ ΠΑΡΑΜΕΤΡΩΝ':>22} | {'ΕΚΠΑΙΔΕΥΣΙΜΕΣ':>22} | (M)")
    print("=" * 80)

    # --- R3D-18 ---
    try:
        r3d = torchvision.models.video.r3d_18(weights="DEFAULT")
        r3d.fc = nn.Linear(r3d.fc.in_features, NUM_CLASSES)
        report("R3D-18", r3d)
    except Exception as e:
        print(f"R3D-18 — ΣΦΑΛΜΑ: {e}")

    # --- MC3-18 ---
    try:
        mc3 = torchvision.models.video.mc3_18(weights="DEFAULT")
        mc3.fc = nn.Linear(mc3.fc.in_features, NUM_CLASSES)
        report("MC3-18", mc3)
    except Exception as e:
        print(f"MC3-18 — ΣΦΑΛΜΑ: {e}")

    # --- TimeSformer ---
    try:
        timesformer = my_models.TimeSformerDFEW(num_classes=NUM_CLASSES, pretrained=True)
        report("TimeSformer", timesformer)
    except Exception as e:
        print(f"TimeSformer — ΣΦΑΛΜΑ: {e}")

    # --- VideoMAE ---
    try:
        videomae = my_models.VideoMAEDFEW(num_classes=NUM_CLASSES, pretrained=True)
        report("VideoMAE", videomae)
    except Exception as e:
        print(f"VideoMAE — ΣΦΑΛΜΑ: {e}")

    # --- Former-DFER ---
    try:
        former = my_models.FormerDFER_Wrapper(num_classes=NUM_CLASSES)
        report("Former-DFER", former)
    except Exception as e:
        print(f"Former-DFER — ΣΦΑΛΜΑ: {e}")

    # --- Υβριδικό ---
    try:
        hybrid = my_model.CNN_Transformer(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
        report("Hybrid (CNN-Trans)", hybrid)
    except Exception as e:
        print(f"Hybrid — ΣΦΑΛΜΑ: {e}")

    print("=" * 80)
