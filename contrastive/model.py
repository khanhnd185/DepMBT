import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from transformer import *

def get_projection(input_dim, output_dim, projection_type):
    if projection_type == 'minimal':
        return nn.Linear(input_dim, output_dim)
    if projection_type == 'conv1d':
        return nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, bias=False)
    elif projection_type == '':
        return nn.Identity()
    else:
        raise NotImplementedError

class MBT(nn.Module):
    def __init__(self, v_dim, a_dim, embed_dim, num_bottle_token=4, bottle_layer=1
                , project_type='minimal', num_head=8, drop=.1, num_layers=4, feat_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.bottle_layer = bottle_layer
        self.num_bottle_token = num_bottle_token
        self.project_type_conv1d = (project_type=='conv1d')
        
        self.audio_prj = get_projection(a_dim, embed_dim, project_type)
        self.video_prj = get_projection(v_dim, embed_dim, project_type)

        ff = embed_dim
        layer = EncoderLayer(embed_dim, num_head, ff, drop)
        self.video_layers = clones(layer, num_layers)
        self.audio_layers = clones(layer, num_layers)

        self.mask_cls = nn.Parameter(torch.ones(1, 1))
        self.mask_bot = nn.Parameter(torch.ones(1, num_bottle_token))
        self.bot_token = nn.Parameter(torch.zeros(1, num_bottle_token, embed_dim))
        self.acls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vcls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norma = LayerNorm(layer.size)
        self.normv = LayerNorm(layer.size)
        self.head = nn.Linear(embed_dim, feat_dim)

        trunc_normal_(self.bot_token, std=.02)
        trunc_normal_(self.acls_token, std=.02)
        trunc_normal_(self.vcls_token, std=.02)

    def forward(self, a, v, m):
        '''
            a : (batch_size, seq_len, a_dim)
            v : (batch_size, seq_len, v_dim)
        '''
        B = a.shape[0]
        if self.project_type_conv1d:
            a = self.audio_prj(a.transpose(1, 2)).transpose(1, 2)
            v = self.video_prj(v.transpose(1, 2)).transpose(1, 2)
        else:
            a = self.audio_prj(a)
            v = self.video_prj(v)

        
        acls_tokens = self.acls_token.expand(B, -1, -1)
        a = torch.cat((acls_tokens, a), dim=1)
        
        vcls_tokens = self.vcls_token.expand(B, -1, -1)
        v = torch.cat((vcls_tokens, v), dim=1)

        mask_cls = self.mask_cls.expand(B, -1)
        mask = torch.cat((mask_cls, m), dim=1)

        for i in range(self.bottle_layer):
            v = self.video_layers[i](v, mask)
            a = self.audio_layers[i](a, mask)

        mask_bot = self.mask_bot.expand(B, -1)
        mask = torch.cat((mask_bot, mask), dim=1)
        bot_token = self.bot_token.expand(B, -1, -1)

        for i in range(self.bottle_layer, self.num_layers):
            a = torch.cat((bot_token, a), dim=1)
            v = torch.cat((bot_token, v), dim=1)

            v = self.video_layers[i](v, mask)
            a = self.audio_layers[i](a, mask)

            bot_token = (a[:, :self.num_bottle_token] + v[:, :self.num_bottle_token]) / 2
            a = a[:, self.num_bottle_token:]
            v = v[:, self.num_bottle_token:]

        a = self.norma(a)
        v = self.normv(v)
        feat = (a[:, 0] + v[:, 0]) / 2

        return feat

class SupConMBT(nn.Module):
    """backbone + projection head"""
    def __init__(self, v_dim, a_dim, embed_dim, head='mlp', feat_dim=128):
        super(SupConMBT, self).__init__()
        self.encoder = MBT(v_dim, a_dim, embed_dim)
        if head == 'linear':
            self.head = nn.Linear(embed_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, a, v, m):
        feat = self.encoder(a, v, m)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class CEMBT(nn.Module):
    """backbone + projection head"""
    def __init__(self, v_dim, a_dim, embed_dim, head='mlp', num_classes=2):
        super(CEMBT, self).__init__()
        self.encoder = MBT(v_dim, a_dim, embed_dim)
        if head == 'linear':
            self.head = nn.Linear(embed_dim, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, a, v, m):
        feat = self.encoder(a, v, m)
        feat = self.head(feat)
        return feat

class EarlyConcat(nn.Module):
    def __init__(self, v_dim, a_dim, embed_dim, project_type='minimal', num_layers=4,
                num_heads=8, head='mlp', num_classes=2, drop=0.1, feed_forward=256):
        super(EarlyConcat, self).__init__()
        
        self.audio_prj = get_projection(a_dim, embed_dim, project_type)
        self.video_prj = get_projection(v_dim, embed_dim, project_type)
        self.project_type_conv1d = (project_type=='conv1d')

        self.encoder = Encoder(embed_dim, num_heads, feed_forward, drop, num_layers)
        if head == 'linear':
            self.head = nn.Linear(embed_dim, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.mask_cls = nn.Parameter(torch.ones(1, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        trunc_normal_(self.cls_token, std=.02)

    def forward(self, a, v, m):
        B = a.shape[0]
        if self.project_type_conv1d:
            a = self.audio_prj(a.transpose(1, 2)).transpose(1, 2)
            v = self.video_prj(v.transpose(1, 2)).transpose(1, 2)
        else:
            a = self.audio_prj(a)
            v = self.video_prj(v)

        feat = torch.cat((a, v), dim=1)
        mask = torch.cat((m, m), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls_tokens, feat), dim=1)

        mask_cls = self.mask_cls.expand(B, -1)
        mask = torch.cat((mask_cls, mask), dim=1)

        feat = self.encoder(feat, mask)
        out = self.head(feat[:, 0])
        return out


class MS2OS(nn.Module):
    def __init__(self, v_dim, a_dim, embed_dim, project_type='minimal', num_layers=2,
                num_heads=4, head='mlp', num_classes=2, drop=0.1, feed_forward=256):
        super(MS2OS, self).__init__()
        
        self.audio_prj = get_projection(a_dim, embed_dim, project_type)
        self.video_prj = get_projection(v_dim, embed_dim, project_type)
        self.project_type_conv1d = (project_type=='conv1d')

        self.video_encoder = Encoder(embed_dim, num_heads, feed_forward, drop, num_layers)
        self.audio_encoder = Encoder(embed_dim, num_heads, feed_forward, drop, num_layers)
        self.fused_encoder = Encoder(embed_dim, num_heads, feed_forward, drop, num_layers)

        if head == 'linear':
            self.head = nn.Linear(embed_dim, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.mask_cls = nn.Parameter(torch.ones(1, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        trunc_normal_(self.cls_token, std=.02)

    def forward(self, a, v, m):
        B = a.shape[0]
        if self.project_type_conv1d:
            a = self.audio_prj(a.transpose(1, 2)).transpose(1, 2)
            v = self.video_prj(v.transpose(1, 2)).transpose(1, 2)
        else:
            a = self.audio_prj(a)
            v = self.video_prj(v)

        a = self.audio_encoder(a, m)
        v = self.video_encoder(v, m)

        feat = torch.cat((a, v), dim=1)
        mask = torch.cat((m, m), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls_tokens, feat), dim=1)

        mask_cls = self.mask_cls.expand(B, -1)
        mask = torch.cat((mask_cls, mask), dim=1)

        feat = self.fused_encoder(feat, mask)
        out = self.head(feat[:, 0])
        return out

class CrossAttention(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension, project_type='minimal', pre_norm=True,
                head = 'mlp', feed_forward = 256, dropout = 0.1, num_heads = 8, num_classes = 2):
        super().__init__()
        self.pre_norm = pre_norm
        self.project_type_conv1d = (project_type == 'conv1d')
        self.audio_prj = get_projection(audio_dimension, fused_dimension, project_type)
        self.video_prj = get_projection(video_dimension, fused_dimension, project_type)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fused_dimension))
        self.mask_appnd = nn.Parameter(torch.ones(1, 1))
        trunc_normal_(self.cls_token, std=.02)

        norms = []
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        self.norms = nn.ModuleList(norms)

        self.afeed_forward = PositionwiseFeedForward(fused_dimension, feed_forward, dropout)
        self.adrop1 = nn.Dropout(dropout)
        self.adrop2 = nn.Dropout(dropout)

        self.vfeed_forward = PositionwiseFeedForward(fused_dimension, feed_forward, dropout)
        self.vdrop1 = nn.Dropout(dropout)
        self.vdrop2 = nn.Dropout(dropout)

        self.audio_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.video_attn = MultiHeadedAttention(num_heads, fused_dimension)

        self.fused_encoder = Encoder(fused_dimension, num_heads, feed_forward, dropout, 3)
        if head == 'linear':
            self.head = nn.Linear(fused_dimension, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(fused_dimension, fused_dimension),
                nn.ReLU(inplace=True),
                nn.Linear(fused_dimension, num_classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def maybe_normalize(self, i, x, pre):
        if self.pre_norm == pre:
            return  self.norms[i](x)

        return x

    def forward(self, a, v, m):
        B = a.shape[0]
        if self.project_type_conv1d:
            a = self.audio_prj(a.transpose(1, 2)).transpose(1, 2)
            v = self.video_prj(v.transpose(1, 2)).transpose(1, 2)
        else:
            a = self.audio_prj(a)
            v = self.video_prj(v)

        residual_a = a
        residual_v = v
        a = self.maybe_normalize(0, a, pre=True)
        v = self.maybe_normalize(1, v, pre=True)
        a = residual_a + self.adrop1(self.audio_attn(a, v, v, m))
        v = residual_v + self.vdrop2(self.video_attn(v, a, a, m))
        a = self.maybe_normalize(0, a, pre=False)
        v = self.maybe_normalize(1, v, pre=False)

        residual = a
        a = self.maybe_normalize(2, a, pre=True)
        a = residual + self.adrop2(self.afeed_forward(a))
        a = self.maybe_normalize(2, a, pre=False)

        residual = v
        v = self.maybe_normalize(3, v, pre=True)
        v = residual + self.vdrop2(self.vfeed_forward(v))
        v = self.maybe_normalize(3, v, pre=False)

        f = torch.cat((a, v), dim=1)
        m = torch.cat((m, m), dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_appnd = self.mask_appnd.expand(B, -1)
        m = torch.cat((mask_appnd, m), dim=1)
        f = torch.cat((cls_tokens, f), dim=1)


        feat = self.fused_encoder(f, m)
        out = self.head(feat[:, 0])

        return out

class FullAttention(nn.Module):
    def __init__(self, video_dimension, audio_dimension, fused_dimension, project_type='minimal', pre_norm=True,
                head = 'mlp', feed_forward = 256, dropout = 0.1, num_heads = 8, num_classes = 2,):
        super().__init__()
        self.pre_norm = pre_norm
        self.project_type_conv1d = (project_type == 'conv1d')
        self.audio_prj = get_projection(audio_dimension, fused_dimension, project_type)
        self.video_prj = get_projection(video_dimension, fused_dimension, project_type)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fused_dimension))
        self.mask_appnd = nn.Parameter(torch.ones(1, 1))
        trunc_normal_(self.cls_token, std=.02)

        norms = []
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        norms.append(LayerNorm(fused_dimension))
        self.norms = nn.ModuleList(norms)

        self.aself_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.afeed_forward = PositionwiseFeedForward(fused_dimension, feed_forward, dropout)
        self.adrop1 = nn.Dropout(dropout)
        self.adrop2 = nn.Dropout(dropout)

        self.vself_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.vfeed_forward = PositionwiseFeedForward(fused_dimension, feed_forward, dropout)
        self.vdrop1 = nn.Dropout(dropout)
        self.vdrop2 = nn.Dropout(dropout)

        self.audio_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.video_attn = MultiHeadedAttention(num_heads, fused_dimension)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.fused_encoder = Encoder(fused_dimension, num_heads, feed_forward, dropout, 3)
        if head == 'linear':
            self.head = nn.Linear(fused_dimension, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(fused_dimension, fused_dimension),
                nn.ReLU(inplace=True),
                nn.Linear(fused_dimension, num_classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def maybe_normalize(self, i, x, pre):
        if self.pre_norm == pre:
            return  self.norms[i](x)

        return x

    def forward(self, a, v, m):
        B = a.shape[0]
        if self.project_type_conv1d:
            a = self.audio_prj(a.transpose(1, 2)).transpose(1, 2)
            v = self.video_prj(v.transpose(1, 2)).transpose(1, 2)
        else:
            a = self.audio_prj(a)
            v = self.video_prj(v)

        residual = a
        a = self.maybe_normalize(0, a, pre=True)
        a = residual + self.adrop1(self.aself_attn(a, a, a, m))
        a = self.maybe_normalize(0, a, pre=False)

        residual = a
        a = self.maybe_normalize(1, a, pre=True)
        a = residual + self.adrop2(self.afeed_forward(a))
        a = self.maybe_normalize(1, a, pre=False)

        residual = v
        v = self.maybe_normalize(2, v, pre=True)
        v = residual + self.vdrop1(self.vself_attn(v, v, v, m))
        v = self.maybe_normalize(2, v, pre=False)

        residual = v
        v = self.maybe_normalize(3, v, pre=True)
        v = residual + self.vdrop2(self.vfeed_forward(v))
        v = self.maybe_normalize(3, v, pre=False)

        residual_a = a
        residual_v = v
        a = self.maybe_normalize(4, a, pre=True)
        v = self.maybe_normalize(5, v, pre=True)
        a = residual_a + self.drop1(self.audio_attn(a, v, v, m))
        v = residual_v + self.drop2(self.video_attn(v, a, a, m))
        a = self.maybe_normalize(4, a, pre=False)
        v = self.maybe_normalize(5, v, pre=False)

        f = torch.cat((a, v), dim=1)
        m = torch.cat((m, m), dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_appnd = self.mask_appnd.expand(B, -1)
        m = torch.cat((mask_appnd, m), dim=1)
        f = torch.cat((cls_tokens, f), dim=1)


        feat = self.fused_encoder(f, m)
        out = self.head(feat[:, 0])

        return out

