import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention

from layers import FusionBlock, get_projection, GAP
from annotated_transformer import Encoder, Decoder, LayerNorm
from annotated_transformer import MultiHeadedAttention, PositionwiseFeedForward

from detr_transformer import TransformerEncoder, TransformerEncoderLayer

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
        dropout = 0.1
        num_layers = 1
        num_heads = 1

        self.audio_prj = get_projection(audio_dimension, fused_dimension, 'minimal')
        self.video_prj = get_projection(video_dimension, fused_dimension, 'minimal')
        self.audio_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
        self.video_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
        self.fused_dec = Decoder(fused_dimension, num_heads, feed_forward, dropout)
        self.gap = GAP()
        self.mlp = ShallowNN(fused_dimension*2, hidden_features=fused_dimension, out_features=1, drop=dropout)
    
    def forward(self, a, v, m):
        a = self.audio_prj(a)
        v = self.video_prj(v)
        a = self.audio_enc(a, m)
        v = self.video_enc(v, m)
        f = self.fused_dec(a, v, m)
        out = self.gap(f, m)
        out = self.mlp(out).squeeze(-1)

        return out




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

class AblationModel(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension, config_num):
        super().__init__()
        feed_forward = 256
        dropout = 0.2
        num_layers = 1
        num_heads = 1

        self.audio_prj = get_projection(audio_dimension, fused_dimension, 'minimal')
        self.video_prj = get_projection(video_dimension, fused_dimension, 'minimal')

        self.enable_self_attention = config_num % 2
        self.enable_cross_attention = (config_num // 2) % 2
        self.enable_fused_attention = (config_num // 4) % 2


        self.aself_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.afeed_forward = PositionwiseFeedForward(fused_dimension, feed_forward, dropout)
        self.anorm1 = LayerNorm(fused_dimension)
        self.anorm2 = LayerNorm(fused_dimension)
        self.adrop1 = nn.Dropout(dropout)
        self.adrop2 = nn.Dropout(dropout)
        self.anorm3 = LayerNorm(fused_dimension)

        self.vself_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.vfeed_forward = PositionwiseFeedForward(fused_dimension, feed_forward, dropout)
        self.vnorm1 = LayerNorm(fused_dimension)
        self.vnorm2 = LayerNorm(fused_dimension)
        self.vdrop1 = nn.Dropout(dropout)
        self.vdrop2 = nn.Dropout(dropout)
        self.vnorm3 = LayerNorm(fused_dimension)

        self.audio_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)
        self.video_enc = Encoder(fused_dimension, num_heads, feed_forward, dropout, num_layers)

        self.audio_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.video_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.norm1 = LayerNorm(fused_dimension)
        self.norm2 = LayerNorm(fused_dimension)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.fused_attn = MultiHeadedAttention(num_heads, fused_dimension*2)
        self.feed_forward = PositionwiseFeedForward(fused_dimension*2, feed_forward, dropout)
        self.norm3 = LayerNorm(fused_dimension*2)
        self.norm4 = LayerNorm(fused_dimension*2)
        self.drop3 = nn.Dropout(dropout)
        self.drop4 = nn.Dropout(dropout)

        self.gap = GAP()
        self.mlp = ShallowNN(fused_dimension*2, hidden_features=fused_dimension, out_features=1, drop=dropout)

    def forward(self, a, v, m):
        a = self.audio_prj(a)
        v = self.video_prj(v)
        if self.enable_self_attention:
            ax1 = self.anorm1(a)
            a = a + self.adrop1(self.aself_attn(ax1, ax1, ax1, m))
            ax2 = self.anorm2(a)
            a = a + self.adrop2(self.afeed_forward(ax2))
            a = self.anorm3(a)

            vx1 = self.vnorm1(v)
            v = v + self.vdrop1(self.vself_attn(vx1, vx1, vx1, m))
            vx2 = self.vnorm2(v)
            v = v + self.vdrop2(self.vfeed_forward(vx2))
            v = self.vnorm3(v)

        if self.enable_cross_attention:
            a1 = a + self.drop1(self.audio_attn(self.norm1(a), v, v, m))
            v1 = v + self.drop2(self.video_attn(self.norm2(v), a, a, m))
        else:
            a1 = a
            v1 = v

        f = torch.cat((a1, v1), dim=2)

        if self.enable_fused_attention:
            f1 = self.norm3(f)
            f = f + self.drop3(self.fused_attn(f1, f1, f1, m))
            f2 = self.norm4(f)
            f = f + self.drop4(self.feed_forward(f2))

        out = self.gap(f, m)
        out = self.mlp(out).squeeze(-1)

        return out


class DetrTransformerFusion(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension):
        super().__init__()
        feed_forward = 256
        dropout = 0.2
        num_layers = 1
        num_heads = 1
        normalize_before = True

        self.audio_prj = get_projection(audio_dimension, fused_dimension, 'minimal')
        self.video_prj = get_projection(video_dimension, fused_dimension, 'minimal')

        
        a_encoder_layer = TransformerEncoderLayer(fused_dimension, num_heads, feed_forward, dropout, "relu", normalize_before)
        a_encoder_norm = nn.LayerNorm(fused_dimension) if normalize_before else None
        self.audio_enc = TransformerEncoder(a_encoder_layer, num_layers, a_encoder_norm)

        v_encoder_layer = TransformerEncoderLayer(fused_dimension, num_heads, feed_forward, dropout, "relu", normalize_before)
        v_encoder_norm = nn.LayerNorm(fused_dimension) if normalize_before else None
        self.video_enc = TransformerEncoder(v_encoder_layer, num_layers, v_encoder_norm)

        f_encoder_layer = TransformerEncoderLayer(fused_dimension*2, num_heads, feed_forward, dropout, "relu", normalize_before)
        f_encoder_norm = nn.LayerNorm(fused_dimension*2) if normalize_before else None
        self.fused_enc = TransformerEncoder(f_encoder_layer, num_layers, f_encoder_norm)

        self.gap = GAP()
        self.mlp = ShallowNN(fused_dimension*2, hidden_features=fused_dimension, out_features=1, drop=dropout)
    
    def forward(self, a, v, m):
        mask = (m == 0).long()
        a = self.audio_prj(a)
        v = self.video_prj(v)
        a = a.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        a = self.audio_enc(a, src_key_padding_mask=mask)
        v = self.video_enc(v, src_key_padding_mask=mask)
        f = torch.cat((a, v), dim=2)
        f = self.fused_enc(f, src_key_padding_mask=mask)
        f = f.permute(1, 0, 2)
        out = self.gap(f, m)
        out = self.mlp(out).squeeze(-1)

        return out


