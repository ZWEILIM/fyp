
import torch
import torch.nn as nn
import torch.nn.functional as F
import mictresnet


__all__ = ['MiCTRANet', 'init_lstm_hidden']


class MiCTRANet(nn.Module):
    def __init__(self, backbone, hidden_size, attn_size, output_size,
                 pretrained=False, mode='online'):
        super(MiCTRANet, self).__init__()

        if mode not in ('online', 'offline'):
            raise ValueError('Invalid value for parameter `mode`: {}'.format(mode))
        self.mode = mode
        self.cnn = mictresnet.get_mictresnet(backbone, no_top=True, pretrained=pretrained)
        self.attn_cell = VisualAttentionCell(hidden_size, attn_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, imgs, h0, prior_maps, feat_idx=None):

        feat_maps = self.cnn(imgs)
        B, C, L, _, _ = list(feat_maps.size())
        feat_maps = feat_maps.permute(2, 0, 3, 4, 1).view([L, B, -1, C]).transpose(1, 2)

        out_lstm = []
        h = h0
        if self.mode == 'online':
            i = (L // 2) if feat_idx is None else feat_idx
            prior_maps = prior_maps.transpose(1, 0).view([1, B, -1]).transpose(1, 2)
            h = self.attn_cell(h, feat_maps[i], prior_maps[0])
            out_lstm.append(h[0])
        else:
            prior_maps = prior_maps.transpose(1, 0).view([L, B, -1]).transpose(1, 2)
            for i in range(L):
                h = self.attn_cell(h, feat_maps[i], prior_maps[i])
                out_lstm.append(h[0])
        logits = self.fc(torch.stack(out_lstm, 0))
        probs = F.softmax(logits, dim=-1).transpose(0, 1)
        return probs, h


class VisualAttentionCell(nn.Module):
    def __init__(self, hidden_size, attn_size):
        super(VisualAttentionCell, self).__init__()
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.lstm_cell = nn.LSTMCell(attn_size, hidden_size)
        self.v = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.Wh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.Wk = nn.Parameter(torch.zeros(attn_size, hidden_size))
        self.Wv = nn.Parameter(torch.zeros(attn_size, hidden_size))

    def forward(self, hidden, feat_map, prior_map):

        H = self.hidden_size
        N, B, C = list(feat_map.size())  # number of pixels, batch size and number of channels

        query = torch.matmul(hidden[0], self.Wh)
        key = torch.matmul(feat_map.view(-1, C), self.Wk).view(N, B, H)
        value = torch.matmul(feat_map.view(-1, C), self.Wv).view(N, B, C)

        scores = torch.tanh(query + key).view(-1, H)
        scores = torch.matmul(scores, self.v).view(N, B)
        scores = F.softmax(scores, dim=0)
        attn_weights = scores * prior_map

        context = (attn_weights.view(N, B, 1).repeat(1, 1, C) * value).sum(dim=0)  # [B, C]
        sum_weights = attn_weights.sum(dim=0).view(B, 1).clamp(min=1.0e-5)  # [B, 1]
        return self.lstm_cell(context / sum_weights, hidden)


def init_lstm_hidden(batch_size, hidden_size, dtype=torch.float, device=torch.device('cuda')):
    return (torch.zeros((batch_size, hidden_size), dtype=dtype, device=device),
            torch.zeros((batch_size, hidden_size), dtype=dtype, device=device))
