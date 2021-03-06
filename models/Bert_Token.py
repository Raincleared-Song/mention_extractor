import torch
import torch.nn as nn
from configs import ConfigBase
from utils import ParallelCollector
from transformers import AutoModelForTokenClassification


class BertTokenClassification(nn.Module):
    def __init__(self, config: ConfigBase):
        super(BertTokenClassification, self).__init__()
        self.config = config
        self.bert = AutoModelForTokenClassification.from_pretrained(config.bert_path, num_labels=config.label_num)
        self.pad_label_id = -100

    def forward(self,
                input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                lengths=None, labels=None, flags=None, rank=None):

        if 'bert' in self.config.bert_path:
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
        elif 't5' in self.config.bert_path:
            prediction = self.bert.encoder(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )[0]
        else:
            prediction = self.bert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )[0]

        _, prediction = torch.max(prediction, -1)

        res_labels = self.normalize(prediction, flags, lengths)
        true_labels = self.normalize(labels, flags, lengths)

        if self.config.n_gpu > 1:
            cur_rank = int(rank.item())
        else:
            cur_rank = 0
        ParallelCollector.label_container[cur_rank] = res_labels
        ParallelCollector.true_container[cur_rank] = true_labels
        ParallelCollector.predict_container[cur_rank] = prediction.cpu().tolist()

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
                    result.append(self.config.id2label[logit[i]])
            results.append(result)
            assert len(result) == sum(flag)
        return results
