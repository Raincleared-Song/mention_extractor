import json
from .config_base import ConfigBase


class ConfigSupervised(ConfigBase):
    """信息配置类"""
    # 数据信息
    # data_path = {
    #     'train': 'data/supervised/train.txt',
    #     'valid': 'data/supervised/dev.txt',
    #     'test': 'data/supervised/test.txt'
    # }
    data_path = {
        'train': 'data/episode-data/intra/train_10_1.txt',
        'valid': 'data/episode-data/intra/dev_10_1.txt',
        'test': 'data/episode-data/intra/test_10_1.txt'
    }

    label_map = {
        'type': 'data/labels.json',
        'type_bio': 'data/labels_bio.json',
        'mention': 'data/labels_men.json',
        'mention_bio': 'data/labels_men_bio.json',
    }
    label_type = 'mention_bio'
    overwrite_cache = False

    __fin__ = open(label_map[label_type], encoding='utf-8')
    id2label = json.load(__fin__) + ['[CLS]', '[SEP]', 'X']
    __fin__.close()
    negative_labels = ['O', '[CLS]', '[SEP]', 'X', 'None']

    label_num = len(id2label)
    label2id = {lab: i for i, lab in enumerate(id2label)}
    assert len(id2label) == len(label2id)

    per_gpu_batch_size = {
        'train': 16,
        'valid': 16,
        'test': 16,
    }

    # 使用的方法/输出目录名
    output_path = 'checkpoint'
    model_name = 'BERT-Crf'
    model_path = 'fewnerd-mention_bio-bert_crf-intra1001'
    assert model_name in ['BERT-Crf', 'BERT-BiLSTM-Crf', 'Bert-Token-Classification']

    # 句子最大长度
    max_seq_length = 128

    # 训练设备
    main_device = 'cuda:0'

    # Crf的默认训练参数
    optimizer = 'adamw'
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    weight_decay = 0.0

    num_warmup_steps = 0
    lstm_hidden_size = 512

    max_step = -1
    num_epoch = 10
    max_grad_norm = 1.0
    grad_accu_step = 1  # 梯度累加
    save_step = 2000  # 每 2000 步保存一次
    save_epoch = 1   # 每 1 轮保存一次