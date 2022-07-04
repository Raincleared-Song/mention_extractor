import json
from .config_base import ConfigBase


class ConfigFewshot(ConfigBase):
    """信息配置类"""
    task = 'fewshot'

    part = 'intra'
    n_way = 10
    n_shot = 1

    data_path = f'data/episode-data/{part}/test_{n_way}_{n_shot}.jsonl'

    label_map = {
        'type': 'data/labels.json',
        'type_bio': 'data/labels_bio.json',
        'mention': 'data/labels_men.json',
        'mention_bio': 'data/labels_men_bio.json',
    }
    label_type = 'mention_bio'
    overwrite_cache = False

    per_gpu_batch_size = {
        'train': 8,
        'valid': 8,
        'test': 8,
    }

    __fin__ = open(label_map[label_type], encoding='utf-8')
    id2label = json.load(__fin__) + ['[CLS]', '[SEP]', 'X']
    __fin__.close()
    negative_labels = ['O', '[CLS]', '[SEP]', 'X', 'None']

    label_num = len(id2label)
    label2id = {lab: i for i, lab in enumerate(id2label)}
    assert len(id2label) == len(label2id)

    # 使用的方法/输出目录名
    output_path = 'checkpoint'
    model_name = 'BERT-Crf'
    model_path = 'fewnerd-fewshot-mention_bio-bert_crf-{0}{1:02}{2:02}_finetune_200'
    assert model_name in ['BERT-Crf', 'BERT-BiLSTM-Crf', 'Bert-Token-Classification']

    max_seq_length_map = {
        (5, 5): 32, (10, 5): 32, (5, 1): 64, (10, 1): 64,
    }
    # 句子最大长度
    max_seq_length = max_seq_length_map[(n_way, n_shot)]

    # 训练设备
    main_device = 'cuda:0'

    # Crf的默认训练参数
    optimizer = 'adamw'
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    weight_decay = 0.0

    num_warmup_steps = -1
    lstm_hidden_size = 512

    max_step = -1
    num_epoch = 10
    skip_trained_steps = False  # 是否跳过已训练轮
    max_grad_norm = 1.0
    grad_accu_step = 1  # 梯度累加
    save_step = -1  # 每 2000 步保存一次
    save_epoch = 1   # 每 1 轮保存一次
    save_model = False
