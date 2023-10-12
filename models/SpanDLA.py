from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AdamW
from models.DynamicWindowAttention import DynamicAttention
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchcrf import CRF
from sentence_transformers import SentenceTransformer
from utils import maybe_cuda, setup_logger, unsort
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.mlp import MLP
from models.MatrixLoss import MatrixLoss

logger = setup_logger(__name__, 'train.log')


# profilerLogger = setup_logger("profilerLogger", 'profiler.log', True)
# SEED = 0
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# np.random.seed(SEED)


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden))), \
           Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden)))


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, num_layers=2, hidden=200, labels_num=2, dropout=0.1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.dropout = dropout
        self.bert = SentenceTransformer("/dev_data/sxc/data/pubmedbert")
        self.flag = True
        self.sentence_lstm = nn.LSTM(input_size=768,
                                     hidden_size=self.hidden,
                                     num_layers=self.num_layers,
                                     batch_first=True,
                                     dropout=dropout,
                                     bidirectional=True)

        self.sentence_sc_lstm = nn.LSTM(input_size=self.hidden * 4,
                                        hidden_size=self.hidden,
                                        num_layers=self.num_layers,
                                        batch_first=True,
                                        dropout=dropout,
                                        bidirectional=True)
        self.seg_att_window = DynamicAttention(d_model=self.hidden * 2)
        self.sc_att_window = DynamicAttention(d_model=self.hidden * 2)
        self.pos_enc = LearnedPositionEncoding(d_model=self.hidden * 2)
        self.sc_fc = nn.Linear(self.sentence_lstm.hidden_size * 4, labels_num - 1)
        self.sc_crf = CRF(labels_num - 1)
        self.seg_loss_weight = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.att_loss = nn.BCEWithLogitsLoss()
        self.att_loss_weight = nn.Parameter(torch.tensor(0.2))
        self.matrix_loss = MatrixLoss(self.sentence_lstm.hidden_size * 4, labels_num)
        logger.info('lstm hidden:{},dropout:{},layers:{};'.format(self.sentence_lstm.hidden_size,
                                                                  self.sentence_lstm.dropout,
                                                                  self.sentence_lstm.num_layers))

    def loss_weight_transform(self, loss, loss_weight):
        norm1 = 1 / torch.pow(loss_weight, 2) / 2
        bias = torch.log(1 + loss_weight)
        return norm1 * loss + bias

    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_document_length - d_length))  # (1, 1, max_length, )
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, )

    def pack_doc_batch(self, batch_size, doc_sizes, sentence_encoding):
        assert batch_size == len(doc_sizes)
        encoded_documents = []
        cur = 0
        for i in range(batch_size):
            doc_size = doc_sizes[i]
            encoded_documents.append(maybe_cuda(sentence_encoding[cur:cur + doc_size, :]))
            cur = cur + doc_size
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)  # (max_doc_length, batch size, 512)
        return docs_tensor, ordered_doc_sizes, ordered_document_idx

    def pack_target_batch(self, batch_size, doc_sizes, target):
        assert batch_size == len(doc_sizes)
        encoded_documents = []
        cur = 0
        for i in range(batch_size):
            doc_size = doc_sizes[i]
            encoded_documents.append(maybe_cuda(target[cur:cur + doc_size]))
            cur = cur + doc_size
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [torch.cat((d, maybe_cuda(torch.zeros(max_doc_size - d.size(0))))).unsqueeze(0) for d in
                       ordered_documents]
        docs_tensor = torch.cat(padded_docs, 0)  # (max_doc_length, batch size, 512)
        return docs_tensor.permute(1, 0), ordered_doc_sizes, ordered_document_idx

    def pad_att(self, att, length, max_len):
        right_pad = maybe_cuda(torch.zeros(length, max_len - length).float())
        bottom_pad = maybe_cuda(torch.zeros(max_len - length, max_len).float())
        padded_right_att = torch.cat((att, right_pad), 1)
        padded_att = torch.cat((padded_right_att, bottom_pad), 0)
        return padded_att

    def unpack_doc_batch(self, padded_tensor, ordered_doc_sizes, ordered_document_idx):
        outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            outputs.append(padded_tensor[0:doc_len, i, :])
        unsorted_outputs = [outputs[i] for i in unsort(ordered_document_idx)]
        return unsorted_outputs

    def generate_seq_mask(self, max_seq_length, seq_lengths):
        seq_mask = [torch.tensor([[1] * size + [0] * (max_seq_length - size)], dtype=torch.uint8) for size in
                    seq_lengths]
        return maybe_cuda(torch.cat(seq_mask, 0))

    def seg_target(self, target):
        temp = target.clone().detach().tolist()
        left = 0
        for i, x in enumerate(temp):
            if x > 0:
                temp[i] = 2
                temp[left] = 1 if i != left else 3
                left = i + 1
        return torch.tensor(temp).long()

    def att_target(self, targets, doc_sizes):
        cur = 0
        docs_sc_att_targets = []
        doc_target = []
        for size in doc_sizes:
            doc_target.append(targets[cur:cur + size])
            cur = cur + size
        for doc in doc_target:
            x = torch.zeros(doc.size(0), doc.size(0))
            left = 0
            right = 0
            while right < doc.size(0):
                if doc[right] == 1:
                    x[left:right + 1, left:right + 1] = torch.ones((right - left + 1), (right - left + 1))
                    left = right + 1
                right += 1
            docs_sc_att_targets.append(maybe_cuda(x.float()))
        return docs_sc_att_targets

    def forward(self, batch, targets):
        batch_size = len(batch)
        all_sentences = []
        doc_sizes = []
        for doc in batch:
            all_sentences.extend(doc)
            doc_sizes.append(len(doc))
        with torch.no_grad():
            all_sentences_encodings = self.bert.encode(all_sentences, convert_to_tensor=True)
        docs_tensor, ordered_doc_sizes, ordered_document_idx = self.pack_doc_batch(batch_size, doc_sizes,
                                                                                   all_sentences_encodings)
        max_doc_size = np.max(doc_sizes)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=batch_size))
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)
        unsorted_outputs = self.unpack_doc_batch(padded_x, ordered_doc_sizes, ordered_document_idx)
        h_lstm = torch.cat(unsorted_outputs)
        sc_att_output = []
        docs_score_local = []
        for doc in unsorted_outputs:
            score_local = self.sc_att_window(self.pos_enc(doc.unsqueeze(0)).squeeze(0), False, True, True)[2]
            docs_score_local.append(score_local)
            sc_att_output.append(torch.matmul(F.softmax(score_local, dim=1), doc))
        socs_sc_att_targets = self.att_target(targets[0], doc_sizes)
        sc_att_loss = torch.cat(
            [self.att_loss(d, t).unsqueeze(0) for d, t in zip(docs_score_local, socs_sc_att_targets)]).mean()
        losses = []
        doc_target = []
        probs = []
        cur = 0
        for size in doc_sizes:
            doc_target.append(targets[1][cur:cur + size])
            cur = cur + size
        for lstm, att, target in zip(unsorted_outputs, sc_att_output, doc_target):
            prob, loss = self.matrix_loss(torch.cat((lstm, att), 1), target)
            losses.append(loss)
        loss_span = torch.cat(losses).mean()

        y = self.sc_fc(torch.cat((torch.cat(unsorted_outputs), torch.cat(sc_att_output)), 1))
        batch_y, ordered_doc_sizes, ordered_document_idx = self.pack_doc_batch(batch_size, doc_sizes, y)
        mask = self.generate_seq_mask(max_doc_size, ordered_doc_sizes).permute(1, 0)
        batch_sc_target, ordered_doc_sizes, ordered_document_idx = self.pack_target_batch(batch_size, doc_sizes,
                                                                                          targets[1] - maybe_cuda(
                                                                                              torch.tensor(1)))
        loss_sc = -self.sc_crf(batch_y, batch_sc_target.long(), mask, reduction='mean')
        ordered_output = self.sc_crf.decode(batch_y, mask)
        unsort_output = [ordered_output[i] for i in unsort(ordered_document_idx)]
        output_seg = []
        for doc_output in unsort_output:
            output_seg.extend([l + 1 for l in doc_output])

        return output_seg, (loss_sc, 0.2 * (loss_span + sc_att_loss)), output_seg, docs_score_local, socs_sc_att_targets


def create(labels_num):
    model = Model(labels_num=labels_num, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    return model, optimizer
