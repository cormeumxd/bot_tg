import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

state_dict = torch.load('checkpoint.pth', map_location=torch.device('cpu'))


class MyConvNet(nn.Module):
    def __init__(self):
        # вызов конструктора предка
        super().__init__()
        # необходмо заранее знать, сколько каналов у картинки (сейчас = 3),
        # которую будем подавать в сеть, больше ничего
        # про входящие картинки знать не нужно
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(14 * 14 * 128, 128)  # !!!
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # (32,32)
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # (14,14)
        x = self.pool(F.relu(self.conv2(x)))
        # (6,6)
        print(x.shape)
        x = x.view(-1, 14 * 14 * 128)  # !!!
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyConvNet()
model.load_state_dict(state_dict)
model.eval()


def predict(img):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((64, 64))])
    img = transform(img)
    # img = img.permute(2,0,1)
    print(img.shape)
    with torch.no_grad():
        pr = model(img)
        _, pred = torch.max(pr, 1)
        print(pr)
    return pred.item()