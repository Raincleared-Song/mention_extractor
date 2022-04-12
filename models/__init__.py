from .Bert_Crf import BertCrf
from .Bert_Token import BertTokenClassification

name_to_model = {
    'BERT-Crf': BertCrf,
    'BERT-BiLSTM-Crf': BertCrf,
    'Bert-Token-Classification': BertTokenClassification,
}
