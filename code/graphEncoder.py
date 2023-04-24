import torch
from torch import nn
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        if mask is None:
            return input + self.module(input)
        else:
            return input + self.module(input, mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, mask=None):
        h = q
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            #mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            #compatibility = compatibility * mask
            compatibility[mask[None, :, :, :].expand_as(compatibility) == 0] = -1e10
        attn = torch.softmax(compatibility, dim=-1)
        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class PositionWiseFeedforward(nn.Module):
    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embed_dim, bias=True),
        )
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.sub_layers(input)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512, normalization='batch'):
        super(MultiHeadAttentionLayer, self).__init__()
        self.self_attention = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm1 = Normalization(embed_dim, normalization)
        self.positionwise_ff = SkipConnection(PositionWiseFeedforward(embed_dim=embed_dim, feed_forward_dim=feed_forward_hidden))
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x, mask):
        x = self.self_attention(x, mask)
        x = self.norm2(self.positionwise_ff(self.norm1(x)))
        return x


class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, node_dim=7, normalization='batch', feed_forward_hidden=512):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
        for layer in self.layers:
            x = layer(x, mask)
        return x, x.mean(dim=1)