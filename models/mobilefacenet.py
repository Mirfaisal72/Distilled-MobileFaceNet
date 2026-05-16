import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, k, s, p):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, p, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def conv_dw(inp, oup, s):
    return nn.Sequential(
        # DEPTH WISE CONVOLUTION (WORKS FOR EACH CHANNEL AR A TIME)
        nn.Conv2d(inp, inp, kernel_size=3, stride=s, padding=1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        # POINT WISE CONVOLUTION (WORKS FOR 3 CHANNELS BUT POINWISE)
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileFaceNet(nn.Module):
    """
    Lightweight MobileFaceNet-like network for 112x112 RGB inputs.
    Produces 128-D L2-normalized embeddings with optional dropout.
    """

    def __init__(self, embedding_size: int = 128, dropout: float = 0.3):
        super().__init__()

        # Stem
        self.layer1 = conv_bn(3, 32, 3, 2, 1)   # 112 -> 56
        self.layer2 = conv_dw(32, 64, 1)
        self.layer3 = conv_dw(64, 128, 2)       # 56 -> 28
        self.layer4 = conv_dw(128, 128, 1)
        self.layer5 = conv_dw(128, 256, 2)      # 28 -> 14
        self.layer6 = conv_dw(256, 256, 1)
        self.layer7 = conv_dw(256, 512, 2)      # 14 -> 7
        self.layer8 = conv_dw(512, 512, 1)

        self.dropout = nn.Dropout(p=dropout)

        # Head: 7x7 -> 1x1 via conv + avgpool
        self.conv_head = nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=0, bias=False)
        self.bn_head = nn.BatchNorm2d(512)

        # Embedding projection
        self.fc = nn.Linear(512, embedding_size, bias=False)
        self.bn_fc = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.dropout(x)
        x = self.bn_head(self.conv_head(x))
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.bn_fc(self.fc(x)) # (B, embedding_size)
        # L2-normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x


class ClassificationHead(nn.Module):
    """Scaled cosine classification head for identity labels."""

    def __init__(self, embedding_size: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, embeddings):
        # cosine classifier with scale factor
        w = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * (embeddings @ w.t())
        return logits
