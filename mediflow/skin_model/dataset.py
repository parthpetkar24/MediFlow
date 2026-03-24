import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

TRAIN = "skin_dataset/train"
TEST  = "skin_dataset/test"

IMG_SIZE = 224
BATCH    = 16   # small dataset → smaller batch

# ---------- Transforms ----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- Datasets ----------
train_dataset = datasets.ImageFolder(TRAIN, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST,  transform=test_transform)

classes     = train_dataset.classes
num_classes = len(classes)

# ---------- Weighted sampler (handles class imbalance) ----------
targets        = np.array(train_dataset.targets)
class_counts   = np.bincount(targets)
class_weights  = 1.0 / class_counts
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# ---------- Loaders ----------
train_loader = DataLoader(
    train_dataset, batch_size=BATCH,
    sampler=sampler, num_workers=0, pin_memory=False,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH,
    shuffle=False, num_workers=0, pin_memory=False,
)