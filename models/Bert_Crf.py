import torch
import torch.nn as nn
from config import Config
from transformers import BertModel
from utils import ParallelCollector
from .crf_base import CRF, DynamicRNN
from torch.nn.utils.rnn import pad_sequence


class BertCrf(nn.Module):
    def __init__(self):
        super(BertCrf, self).__init__()
        self.bert = BertModel.from_pretrained(Config.bert_path)
        self.dropout = nn.Dropout(0.3)
        self.hidden2tag = nn.Linear(in_features=Config.lstm_hidden_size
                                    if Config.model_name == 'BERT-BiLSTM-Crf' else 768,
                                    out_features=Config.label_num, bias=True)
        self.pad_label_id = -100
        self.crf = CRF(tagset_size=Config.label_num, device=Config.main_device)

        if Config.model_name == 'BERT-BiLSTM-Crf':
            self.rnn = DynamicRNN(768)

    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                lengths=None, labels=None, flags=None, rank=None):

        prediction = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[0]

        prediction = self.dropout(prediction)  # where to put dropout?
        if Config.model_name == 'BERT-BiLSTM-Crf':
            prediction = self.rnn(prediction, lengths)
        prediction = self.hidden2tag(prediction)  # [B, L, N]

        pad_masks = (labels != self.pad_label_id)
        loss_masks = ((attention_mask == 1) & pad_masks)

        crf_labels, crf_masks = self.to_crf_pad(labels, loss_masks)
        crf_logits, _ = self.to_crf_pad(prediction, loss_masks)
        loss = self.crf.neg_log_likelihood(crf_logits, crf_masks, crf_labels)

        masks = (attention_mask == 1)
        crf_logits, crf_masks = self.to_crf_pad(prediction, masks)
        crf_masks = crf_masks.sum(axis=2) == crf_masks.shape[2]
        best_path = self.crf(crf_logits, crf_masks)
        # temp_labels = (torch.ones(loss_masks.shape) * self.pad_label_id).to(torch.long)
        temp_labels = (torch.ones(loss_masks.shape) * self.pad_label_id).to(torch.long)
        prediction = self.unpad_crf(best_path, crf_masks, temp_labels, masks)

        res_labels = self.normalize(prediction, flags, lengths)
        true_labels = self.normalize(labels, flags, lengths)

        if Config.n_gpu > 1:
            cur_rank = int(rank.item())
        else:
            cur_rank = 0
        ParallelCollector.label_container[cur_rank] = res_labels
        ParallelCollector.true_container[cur_rank] = true_labels

        return loss.unsqueeze(0)

    def normalize(self, logits, flags, lengths):
        results = []
        logits = logits.tolist()
        lengths = lengths.tolist()
        assert len(logits) == len(flags) == len(lengths)
        for logit, flag, length in zip(logits, flags, lengths):
            result = []
            for i in range(length):
                if flag[i] == 1:
                    assert logit[i] != self.pad_label_id
                    result.append(Config.id2label[logit[i]])
            results.append(result)
            assert len(result) == sum(flag)
        return results

    def to_crf_pad(self, org_array, org_mask):
        crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
        crf_array = pad_sequence(crf_array, batch_first=True, padding_value=self.pad_label_id)
        crf_pad = (crf_array != self.pad_label_id)
        crf_array[~crf_pad] = 0
        return crf_array, crf_pad

    @staticmethod
    def unpad_crf(returned_array, returned_mask, org_array, org_mask):
        out_array = org_array.clone().detach().to(returned_array.device)
        out_array[org_mask] = returned_array[returned_mask]
        return out_array
