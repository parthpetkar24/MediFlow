import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

TRAIN="skin_dataset/train"
TEST="skin_dataset/test"

IMG_SIZE=224
BATCH=32


from torchvision import transforms

train_transform = transforms.Compose([

transforms.Resize((224,224)),

transforms.RandomHorizontalFlip(),

transforms.RandomRotation(10),

transforms.ColorJitter(
brightness=0.2,
contrast=0.2
),
transforms.ToTensor(),
transforms.Normalize(
[0.485,0.456,0.406],
[0.229,0.224,0.225]
)
])

test_transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(
[0.485,0.456,0.406],
[0.229,0.224,0.225]
)
])

train_dataset=datasets.ImageFolder(
TRAIN,
transform=train_transform
)

test_dataset=datasets.ImageFolder(
TEST,
transform=test_transform
)

train_loader=DataLoader(
train_dataset,
batch_size=BATCH,
shuffle=True
)

test_loader=DataLoader(
test_dataset,
batch_size=BATCH,
shuffle=False
)

classes=train_dataset.classes