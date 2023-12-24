import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_features, out_features, stride):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_features,
                      out_features,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      padding_mode="reflect",
                    ),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.cnn(x)


class Discriminator(nn.Module):

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels,
                      features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_features = features[0]

        for feature in features[1:]:
            layer = CNNBlock(in_features, feature,
                             stride=1 if feature == features[-1] else 2)
            in_features = feature
            layers.append(layer)
        layers.append(nn.Conv2d(in_features, 1, kernel_size=4,
                                padding=1, stride=1, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.rand([32, 3, 256, 256])
    model = Discriminator()
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()