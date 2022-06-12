from typing import Union
from transformers import AutoTokenizer


class ConfigBase:
    """基础信息配置类"""
    data_path: Union[dict, str]

    label_map: dict
    label_type: str
    overwrite_cache: bool

    id2label: list
    negative_labels: list

    label_num: int
    label2id: dict

    per_gpu_batch_size: dict

    output_path: str
    model_name: str
    model_path: str

    # bert 路径
    bert_path = 'bert-base-uncased'
    bert_hidden = 768
    # Dataloader 线程数目
    reader_num = 4
    # 全局切词器
    tokenizer = AutoTokenizer.from_pretrained(bert_path, do_lower_case=True)
    # [unused99] 代表 strip 后为空的词
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused99]"]})
    # 使用 gpu 数量
    n_gpu = 1
    # 随机种子
    seed = 66

    max_seq_length_map: dict
    max_seq_length: int
    main_device: str

    optimizer: str
    learning_rate: float
    adam_epsilon: float
    weight_decay: float

    num_warmup_steps: int
    lstm_hidden_size: int

    max_step: int
    num_epoch: int
    skip_trained_steps: bool
    max_grad_norm: float
    grad_accu_step: int
    save_step: int
    save_epoch: int
    save_model: bool
