import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from utils import maybe_cuda


class DynamicAttention(nn.Module):
    def __init__(self, d_model):
        super(DynamicAttention, self).__init__()
        # dynamic window
        self.query_left_weight = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.key_left_weight = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.query_right_weight = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.key_right_weight = nn.Parameter(torch.FloatTensor(d_model, d_model))
        # self-attention
        self.wq_glob = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.wk_glob = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.wq_loc = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.wk_loc = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self.wv = nn.Parameter(torch.FloatTensor(d_model, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.query_left_weight)
        xavier_uniform_(self.key_left_weight)
        xavier_uniform_(self.query_right_weight)
        xavier_uniform_(self.key_right_weight)
        xavier_uniform_(self.wq_glob)
        xavier_uniform_(self.wk_glob)
        xavier_uniform_(self.wq_loc)
        xavier_uniform_(self.wk_loc)
        xavier_uniform_(self.wv)

    def forward(self, input, is_additive=True, is_return_score=False, is_local=True):
        shape = input.shape  # (sentence,d)
        norm = maybe_cuda(torch.tensor(shape[1]).float().sqrt())
        query_left = torch.matmul(input, self.query_left_weight)
        key_left = torch.matmul(input, self.key_left_weight)
        query_right = torch.matmul(input, self.query_right_weight)
        key_right = torch.matmul(input, self.key_right_weight)
        unit_boundary = maybe_cuda(torch.triu(torch.ones(shape[0], shape[0]), diagonal=1))
        unit_att = maybe_cuda(torch.triu(torch.ones(shape[0], shape[0])))
        left_boundary = F.softmax(torch.matmul(query_left, key_left.transpose(0, 1))
                                  .masked_fill(unit_boundary == 1, float('-inf')) / norm, 1)
        right_boundary = F.softmax(torch.matmul(query_right, key_right.transpose(0, 1))
                                   .masked_fill(unit_boundary.transpose(0, 1) == 1, float('-inf')) / norm, 1)
        att_mask = torch.matmul(left_boundary, unit_att) * torch.matmul(right_boundary, unit_att.transpose(0, 1))

        s_glb = torch.matmul(torch.matmul(input, self.wq_glob),
                             torch.matmul(input, self.wk_glob).transpose(0, 1))
        score = s_glb
        if not is_local:
            att_weight = F.softmax(score, 1)
            att_output = torch.matmul(att_weight, torch.matmul(input, self.wv))
            if is_return_score:
                return att_output, att_weight, score
            return att_output, att_weight
        if is_additive:
            s_loc = torch.matmul(torch.matmul(input, self.wq_loc),
                                 torch.matmul(input, self.wk_loc).transpose(0, 1)) * att_mask
            score += s_loc
        else:
            score *= att_mask
        score /= norm
        att_weight = F.softmax(score, 1)
        att_output = torch.matmul(att_weight, torch.matmul(input, self.wv))
        if is_return_score:
            return att_output, att_weight, score
        return att_output, att_weight
