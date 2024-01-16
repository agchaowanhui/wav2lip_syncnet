import torch
from torch import nn
from torch.nn import functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, inplanes):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        # 通道注意力，即两个全连接层连接
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 16, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=inplanes // 16, out_channels=inplanes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_x = self.max_pool(x)
        avg_x = self.avg_pool(x)
        max_out = self.fc(max_x)
        avg_out = self.fc(avg_x)
        # 最后输出的注意力应该为非负
        out = self.sigmoid(max_out + avg_out)
        return out


class CBAM_Attention(nn.Module):
    def __init__(self, in_channels):
        super(CBAM_Attention, self).__init__()
        self.channel_atten = ChannelAttention(in_channels)
        self.spatial_atten = SpatialAttention()

    def forward(self, x):
        # CBAM attention
        x = self.channel_atten(x) * x
        x = self.spatial_atten(x) * x
        return x


class Conv2dELU_BN_CBAM(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super(Conv2dELU_BN_CBAM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ELU()
        self.residual = residual

        self.cbam_attention = CBAM_Attention(cout)

    def forward(self, x):
        out = self.conv_block(x)

        # CBAM attention
        out = self.cbam_attention(out)

        if self.residual:
            out += x
        return self.act(out)


class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2dELU_BN_CBAM(15, 64, kernel_size=(7, 8), stride=(1, 2), padding=3),  # x/2
            Conv2dELU_BN_CBAM(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2dELU_BN_CBAM(64, 64, kernel_size=7, stride=1, padding=3, residual=True),

            Conv2dELU_BN_CBAM(64, 128, kernel_size=6, stride=2, padding=2),  # x/4
            Conv2dELU_BN_CBAM(128, 128, kernel_size=5, stride=1, padding=2, residual=True),
            Conv2dELU_BN_CBAM(128, 128, kernel_size=5, stride=1, padding=2, residual=True),

            Conv2dELU_BN_CBAM(128, 256, kernel_size=4, stride=2, padding=1),  # x/8
            Conv2dELU_BN_CBAM(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dELU_BN_CBAM(256, 512, kernel_size=4, stride=2, padding=1),  # x/16
            Conv2dELU_BN_CBAM(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dELU_BN_CBAM(512, 512, kernel_size=4, stride=2, padding=1),  # x/32
            Conv2dELU_BN_CBAM(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dELU_BN_CBAM(512, 512, kernel_size=2, stride=2, padding=0),  # x/64
            Conv2dELU_BN_CBAM(512, 512, kernel_size=1, stride=1, padding=0),

            nn.AdaptiveMaxPool2d(1),
        )

        self.audio_encoder = nn.Sequential(
            Conv2dELU_BN_CBAM(1, 64, kernel_size=(9, 5), stride=(5, 1), padding=2),  # 16 16
            Conv2dELU_BN_CBAM(64, 64, kernel_size=5, stride=1, padding=2, residual=True),
            Conv2dELU_BN_CBAM(64, 64, kernel_size=5, stride=1, padding=2, residual=True),

            Conv2dELU_BN_CBAM(64, 128, kernel_size=4, stride=2, padding=1),  # 8
            Conv2dELU_BN_CBAM(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dELU_BN_CBAM(128, 256, kernel_size=4, stride=2, padding=1),  # 4
            Conv2dELU_BN_CBAM(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2dELU_BN_CBAM(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dELU_BN_CBAM(256, 512, kernel_size=2, stride=2, padding=0),  # 2
            Conv2dELU_BN_CBAM(512, 512, kernel_size=1, stride=1, padding=0),

            nn.AdaptiveMaxPool2d(1),
        )

        self.face_fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Sigmoid(),
        )

        self.audio_fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.Sigmoid(),
        )

    def forward(self, audio_sequences, face_sequences, bs):

        audio_embedding = self.audio_encoder(audio_sequences)
        face_embedding = self.face_encoder(face_sequences)

        audio_embedding = audio_embedding.reshape(bs, 512)
        face_embedding = face_embedding.reshape(bs, 512)

        audio_embedding = self.audio_fc(audio_embedding)
        face_embedding = self.face_fc(face_embedding)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding
