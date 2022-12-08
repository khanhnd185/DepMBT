import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention

from layers import FusionBlock

class ShallowNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act2 = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1.weight.data.normal_(0, math.sqrt(1. / hidden_features))
        self.fc2.weight.data.normal_(0, math.sqrt(1. / out_features))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, input_dimension, hidden_features=256, out_features=1):
        super().__init__()
        self.fc = ShallowNN(input_dimension, hidden_features=hidden_features, out_features=out_features)

    def forward(self, feature_audio, feature_video, mask):
        feature_audio, feature_video, mask = feature_audio.sum(dim=1), feature_video.sum(dim=1), mask.sum(dim=1)
        feature_audio = torch.div(feature_audio, mask.unsqueeze(1))
        feature_video = torch.div(feature_video, mask.unsqueeze(1))
        x = torch.cat((feature_audio, feature_video), dim=1)
        x = self.fc(x).squeeze(-1)
        return x

class TransformerFusion(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension):
        super().__init__()
        self.audio = FusionBlock(audio_dimension, 1, mlp_ratio=1)
        self.video = FusionBlock(video_dimension, 1, mlp_ratio=1)
        #self.fused = FusionBlock(fused_dimension, 1)
        self.fc = ShallowNN(161, hidden_features=128, out_features=1)
    
    def forward(self, feature_audio, feature_video, mask):
        feature_audio = self.audio(feature_audio, mask)
        feature_video = self.video(feature_video, mask)
        feature_audio, feature_video, mask = feature_audio.sum(dim=1), feature_video.sum(dim=1), mask.sum(dim=1)
        feature_audio = torch.div(feature_audio, mask.unsqueeze(1))
        feature_video = torch.div(feature_video, mask.unsqueeze(1))
        x = torch.cat((feature_audio, feature_video), dim=1)
        x = self.fc(x).squeeze(-1)

        return x
