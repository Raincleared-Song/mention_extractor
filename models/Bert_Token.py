import torch
import torch.nn as nn
from config import Config
from utils import ParallelCollector
from transformers import BertForTokenClassification


class BertTokenClassification(nn.Module):
    def __init__(self):
        super(BertTokenClassification, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(Config.bert_path, num_labels=Config.label_num)
        self.pad_label_id = -100

    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                lengths=None, labels=None, flags=None, rank=None):

        loss, prediction = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=False,
        )

        _, prediction = torch.max(prediction, -1)

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
        assert logits.dtype not in [torch.float16, torch.float32, torch.float64]
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
