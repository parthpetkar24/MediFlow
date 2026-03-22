import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from dataset import train_loader,test_loader,classes
from model import SkinCNN

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=SkinCNN(len(classes)).to(device)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(
optimizer,
step_size=5,
gamma=0.3
)
EPOCHS=30


for epoch in range(EPOCHS):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(
            outputs,
            labels
        )
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        with torch.no_grad():
            _,pred=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(pred==labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    acc=100*correct/total
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} Accuracy: {acc:.2f}")
    scheduler.step()
torch.save(model.state_dict(),"skin_model/skin_model.pth")

joblib.dump(classes,"skin_model/labels.joblib")
print("Training Complete")