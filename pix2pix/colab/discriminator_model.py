import torch
import torch.nn as nn


class CNNBLock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 2)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=stride, padding=1, bias=False,
                      padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


# DCGAN 输入图像包含两部分，生成图片以及条件图片
class Discriminator(nn.Module):
    def __init__(self, in_channels, features=None):  # 256*256 -> 30*30
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=(4, 4), stride=(2, 2), padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBLock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        self.output = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=(4, 4), stride=(1, 1), padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.initial(x)
        x = self.model(x)
        return self.output(x)


def test():
    x = torch.randn((8, 3, 256, 256))
    y = torch.randn((8, 3, 256, 256))
    model = Discriminator(in_channels=3)
    print(model(x, y).shape)


if __name__ == '__main__':
    test()
