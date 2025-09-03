import torch

from model.lenet import LeNet
from model.model_resnet import ResNet18, ResNet34
from model.model_resnet_official import ResNet50
import torchvision.models as models
import torch.nn as nn


def build_model(args):
    # choose different Neural network model for different args or input
    if args.model == 'lenet':
        netglob = LeNet().to(args.device)

    elif args.model == 'resnet18':
        netglob = ResNet18(args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'resnet34':
        netglob = ResNet34(args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'resnet50':
        netglob = ResNet50(pretrained=False)
        if args.pretrained:
            model = models.resnet50(pretrained=True)
            netglob.load_state_dict(model.state_dict())
        netglob.fc = nn.Linear(2048, args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == "cnn":
        if "mnist" in args.dataset:
            netglob = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "usps" in args.dataset:
            netglob = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)

        elif "cifar10" in args.dataset:
            netglob = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
    else:
        exit('Error: unrecognized model')

    return netglob



class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, get_feature=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        x1 = self.fc1(out)
        out = self.fc(x1)
        if get_feature == True:
            return out, x1
        return out
