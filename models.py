import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention

from layers import FusionBlock, get_projection
from annotated_transformer import Encoder, Decoder
from annotated_transformer import MultiHeadedAttention, PositionwiseFeedForward, clones, SublayerConnection

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

class StanfordTransformerFusion(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension):
        super().__init__()
        feed_forward = 256
        dropout = 0.2
        num_layers = 1
        num_heads = 1

        self.audio_prj = get_projection(audio_dimension, fused_dimension, 'minimal')
        self.video_prj = get_projection(video_dimension, fused_dimension, 'minimal')
        self.audio_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
        self.video_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
        self.fused_dec = Decoder(fused_dimension, num_heads, feed_forward, dropout)
        self.mlp = ShallowNN(fused_dimension*2, hidden_features=fused_dimension, out_features=1, drop=dropout)
    
    def forward(self, a, v, m):
        a = self.audio_prj(a)
        v = self.video_prj(v)
        a = self.audio_enc(a, m)
        v = self.video_enc(v, m)
        f = self.fused_dec(a, v, m)
        f = f.mean(dim=1)
        f = self.mlp(f).squeeze(-1)

        return f



'''
    For Ablation Study
'''
class MaskedPassThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m=None):
        return x

class PassThroughAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, m=None):
        return q

class CustomDecoder(nn.Module):
    def __init__(self, size, h, feed_forward, dropout):
        super(CustomDecoder, self).__init__()
        self.size = size
        self.audio_attn = MultiHeadedAttention(h, size)
        self.video_attn = MultiHeadedAttention(h, size)
        self.fused_attn = MultiHeadedAttention(h, size*2)
        self.feed_forward = PositionwiseFeedForward(size*2, feed_forward, dropout)
        self.sublayer1 = clones(SublayerConnection(size, dropout), 2)
        self.sublayer2 = clones(SublayerConnection(size*2, dropout), 2)

    def forward(self, a, v, mask=None):
        a = self.sublayer1[0](a, lambda a: self.audio_attn(a, v, v, mask))
        v = self.sublayer1[1](v, lambda v: self.video_attn(v, a, a, mask))
        f = torch.cat((a, v), dim=2)
        f = self.sublayer2[0](f, lambda f: self.fused_attn(f, f, f, mask))
        return self.sublayer2[1](f, self.feed_forward)

class AblationModel(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension, config_num):
        super().__init__()
        feed_forward = 256
        dropout = 0.2
        num_layers = 1
        num_heads = 1

        self.audio_prj = get_projection(audio_dimension, fused_dimension, 'minimal')
        self.video_prj = get_projection(video_dimension, fused_dimension, 'minimal')

        self_attention = config_num % 2
        cross_attention = (config_num // 2) % 2
        fused_attention = (config_num // 4) % 2

        if self_attention:
            self.audio_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
            self.video_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
        else:
            self.audio_enc = MaskedPassThrough()
            self.video_enc = MaskedPassThrough()

        if cross_attention:
            self.audio_attn = MultiHeadedAttention(num_heads, fused_dimension)
            self.video_attn = MultiHeadedAttention(num_heads, fused_dimension)
            self.sublayer1 = clones(SublayerConnection(fused_dimension, dropout), 2)
        else:
            self.audio_attn = PassThroughAttention()
            self.video_attn = PassThroughAttention()
            self.sublayer1 = [MaskedPassThrough() for _ in range(2)]

        if fused_attention:
            self.fused_attn = MultiHeadedAttention(num_heads, fused_dimension*2)
            self.feed_forward = PositionwiseFeedForward(fused_dimension*2, feed_forward, dropout)
            self.sublayer2 = clones(SublayerConnection(fused_dimension*2, dropout), 2)
        else:
            self.fused_attn = PassThroughAttention()
            self.feed_forward = nn.Identity()
            self.sublayer2 = [MaskedPassThrough() for _ in range(2)]

        self.mlp = ShallowNN(fused_dimension*2, hidden_features=fused_dimension, out_features=1, drop=dropout)

    def forward(self, a, v, m):
        a = self.audio_prj(a)
        v = self.video_prj(v)
        a = self.audio_enc(a, m)
        v = self.video_enc(v, m)

        a = self.sublayer1[0](a, lambda a: self.audio_attn(a, v, v, m))
        v = self.sublayer1[1](v, lambda v: self.video_attn(v, a, a, m))
        f = torch.cat((a, v), dim=2)
        f = self.sublayer2[0](f, lambda f: self.fused_attn(f, f, f, m))
        f = self.sublayer2[1](f, self.feed_forward)

        f = f.mean(dim=1)
        f = self.mlp(f).squeeze(-1)

        return f
