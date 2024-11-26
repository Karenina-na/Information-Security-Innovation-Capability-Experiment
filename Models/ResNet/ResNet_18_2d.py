import torch
from Models.ResNet.Block.ResBlock_2conv_2d import ResBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_classes = 6
n_channels = 12

class ResBlock(torch.nn.Module):
    """一维残差块，两个卷积层"""
    def __init__(self, In_channel, Med_channel, Out_channel, DownSample=False):
        super(ResBlock, self).__init__()
        self.stride = 1
        if DownSample:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Med_channel, 3, self.stride, padding=1),
            torch.nn.BatchNorm2d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Med_channel, Out_channel, 3, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv2d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=n_channels, classes=n_classes):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool2d(3, 2, 1),

            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            ResBlock(64, 64, 64, False),
            #
            ResBlock(64, 128, 128, True),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            ResBlock(128, 128, 128, False),
            #
            ResBlock(128, 256, 256, True),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            ResBlock(256, 256, 256, False),
            #
            ResBlock(256, 512, 512, True),
            ResBlock(512, 512, 512, False),
            ResBlock(512, 512, 512, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512, classes),
            torch.nn.Dropout(p=0.5, inplace=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifer(x)
        return x


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1].cpu().numpy()
    y_label = torch.max(labels, 1)[1].data.cpu().numpy()
    rights = (pred == y_label).sum()
    return rights, len(labels)


if __name__ == "__main__":
    x = torch.randn(10, 12, 128, 128).to(device)
    model = ResNet().to(device)
    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
