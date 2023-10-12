import torch
from torch import nn
from utils import maybe_cuda, transform2seg, get_mass
from torch.autograd import Variable
import torch.nn.functional as F


class SegmentLengthEmbedding(nn.Embedding):
    def __init__(self, d_model=10, max_len=1000):
        self.d_model = d_model
        self.max_len = max_len
        super(SegmentLengthEmbedding, self).__init__(max_len, d_model)

    def forward(self, x):
        weight = self.weight.data
        return weight[x, :]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ffnn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=True)
        )

    def forward(self, x):
        att_score = self.ffnn(x)
        return att_score


class MatrixLoss(nn.Module):

    def __init__(self, hidden_size, num_tag):
        super(MatrixLoss, self).__init__()
        self.hidden_size = hidden_size
        self.seg_len_emd = SegmentLengthEmbedding()
        self.num_tag = num_tag
        self.fc = nn.Linear(self.hidden_size * 3 + self.seg_len_emd.d_model, num_tag)
        self.attn_layer = AttentionLayer(self.hidden_size)
        self.loss = nn.CrossEntropyLoss()

    def is_same_seg(self, target, i, j):
        return target[j] == target[i] and (target[i:j + 1] == target[j]).all()

    def is_span(self, target, i, j):
        return self.is_same_seg(target, i, j) and ((i - 1 < 0) or target[i - 1] != target[i]) and (
                (j + 1 >= target.shape[0]) or target[j + 1] != target[j])

    def forward(self, hidden, target):
        seq_len = hidden.size(0)
        matrix_hidden = []
        matrix_target = []
        if target.numel() == 1:
            return None, maybe_cuda(torch.tensor([0.]))
        att_score = self.attn_layer(hidden)
        for i in range(seq_len):
            for j in range(i, seq_len):
                if i == j:
                    continue
                att_weight = F.softmax(att_score[i:j + 1, :], 0).transpose(0, 1)
                span = torch.matmul(att_weight, hidden[i:j + 1, :]).squeeze(0)
                matrix_hidden.append(torch.cat((span, hidden[i], hidden[j], self.seg_len_emd(j - i + 1))).unsqueeze(0))
                matrix_target.append(
                    target[j].unsqueeze(0) if self.is_span(target, i, j) else maybe_cuda(torch.tensor([0])))
        matrix_hidden = self.fc(torch.cat(matrix_hidden))
        loss = self.loss(matrix_hidden, maybe_cuda(torch.cat(matrix_target)))
        return matrix_hidden, loss.unsqueeze(0)
