import os
import random
from utils import load_json
from typing import List, Type
from configs import ConfigBase
from torch.utils.data import Dataset


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, proc_words):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.proc_words = proc_words


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, input_mask, segment_ids, label_ids, flags, length, words, extra_labels):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.flags = flags
        self.length = length
        self.words = words
        self.extra_labels = extra_labels


class InputExampleDataset(Dataset):
    def __init__(self, data: List[InputExample]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> InputExample:
        return self.data[item]


def label_convert(label: str, is_begin: bool, config: Type[ConfigBase]) -> str:
    if config.label_type == 'type' or label in config.negative_labels:
        res = label if len(label) <= 2 or label[2:] not in ('B-', 'I-') else label[2:]
    elif config.label_type == 'type_bio':
        res = ('B-' if is_begin else 'I-') + label if len(label) <= 2 or label[2:] not in ('B-', 'I-') else label
    elif config.label_type == 'mention':
        res = 'mention'
    elif config.label_type == 'mention_bio':
        res = ('B-' if is_begin else 'I-') + 'mention'
    else:
        raise ValueError('invalid label type!')
    assert res in config.label2id
    return res


def read_examples_from_file(data_dir: str, prefix: str, config: Type[ConfigBase]) -> List[InputExample]:
    guid_index = 0
    examples = []
    with open(data_dir, encoding="utf-8") as f:
        words = []
        labels = []
        last_label = "O"
        proc_words = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith("-DOCSTART-") or not line.strip():
                if len(words) > 0:
                    assert len(words) == len(labels)
                    examples.append(InputExample(guid="{}-{}".format(prefix, guid_index),
                                                 words=words, labels=labels, proc_words=proc_words))
                    guid_index += 1
                    words = []
                    labels = []
                    last_label = "O"
                    proc_words = []
            else:
                splits = line.split("\t")
                if len(splits) != 2:
                    assert len(splits) == 1
                    splits = ['[unused99]'] + splits
                cur_word = splits[0].strip()
                cur_proc = config.tokenizer.tokenize(cur_word) if len(cur_word) > 0 else []
                if len(cur_word) > 0 and len(cur_proc) > 0:
                    words.append(cur_word)
                    proc_words.append(cur_proc)
                    if len(splits) > 1:
                        cur_label = splits[-1].replace("\n", "")
                    else:
                        # Examples could have no label for mode = "test"
                        cur_label = "O"
                    cur_conv_label = label_convert(cur_label, cur_label != last_label, config)
                    labels.append(cur_conv_label)
                    last_label = cur_label
        if words:
            assert len(words) == len(labels)
            examples.append(InputExample(guid="%s-%d".format(prefix, guid_index),
                                         words=words, labels=labels, proc_words=proc_words))
    return examples


class LargeInputExampleKernel:
    """large dataset for pretraining, singleton"""
    _singleton = None

    config: Type[ConfigBase]
    data_path: str
    file_list: list
    indexes: list
    stats: dict
    total_len: int
    pool_limit: int
    preload_count: int
    cached_samples: list
    cur_progress: int

    def __new__(cls, config: Type[ConfigBase], mode: str):
        if cls._singleton is not None:
            return cls._singleton

        print('Initializing large dataset ......')
        cls.config = config
        assert isinstance(cls.config.data_path, str)
        cls.data_path = config.data_path
        files = load_json(os.path.join(cls.data_path, 'splits.json'))[mode]
        files = [(sub, file) for sub, file in files]
        if mode == 'train':
            random.shuffle(files)
        cls.file_list = files
        stats = load_json(os.path.join(cls.data_path, 'stats.json'))
        stats = {(sub, file): cnt for sub, file, cnt in stats}
        # create indexes
        cls.indexes = [0]
        cur_sum = 0
        for sub, file in files:
            cnt = stats[(sub, file)]
            # assert cnt > cls.config.per_gpu_batch_size[mode]
            cur_sum += cnt
            cls.indexes.append(cur_sum)
        cls.stats = stats
        cls.total_len = cur_sum
        print('loaded file number:', len(cls.file_list))
        print('loaded sample number:', cur_sum)

        # if size larger than 5, remove the earliest one
        cls.pool_limit = 5
        cls.preload_count = 3
        cls.cached_samples = []  # List[fid, examples]
        cls.cur_progress = 0  # the next fid to load
        # preload files
        for _ in range(cls.preload_count):
            cls.push_back(to_pop=False)

        assert cls._singleton is None
        cls._singleton = super(LargeInputExampleKernel, cls).__new__(cls)
        return cls._singleton

    @classmethod
    def push_back(cls, to_pop=True):
        """thread not safe"""
        fid = cls.cur_progress
        if fid in [f for f, _ in cls.cached_samples]:
            return
        sub, file = cls.file_list[fid]
        file_path = os.path.join(cls.data_path, sub, file)
        examples = read_examples_from_file(file_path, f'{sub}+{file}', cls.config)
        cls.cached_samples.append((fid, examples))
        assert len(examples) == cls.stats[(sub, file)]
        cls.cur_progress = (fid + 1) % len(cls.file_list)
        if to_pop and len(cls.cached_samples) > cls.pool_limit:
            cls.pop()

    @classmethod
    def pop(cls):
        """thread not safe"""
        assert len(cls.cached_samples) > 0
        cls.cached_samples = cls.cached_samples[1:]

    @classmethod
    def check_status(cls, next_begin: int, next_end: int, to_push=True):
        """thread not safe"""
        sid, eid = cls.binary_search_fid(next_begin), cls.binary_search_fid(next_end - 1)
        if sid != eid or next_begin == cls.indexes[sid]:
            if to_push:
                cls.push_back()
            return True
        return False

    @classmethod
    def binary_search_fid(cls, did):
        """thread safe"""
        assert did < cls.total_len
        lo, hi = 0, len(cls.indexes)
        while lo < hi:
            mi = (lo + hi) // 2
            if cls.indexes[mi] <= did:
                lo = mi + 1
            else:
                hi = mi
        return lo - 1


class LargeInputExampleDataset(Dataset):
    """wrapper of LargeInputExampleKernel"""
    def __init__(self, config: Type[ConfigBase], mode: str):
        self.kernel = LargeInputExampleKernel(config, mode)

    def __len__(self):
        cls = LargeInputExampleKernel
        return cls.total_len

    def __getitem__(self, item):
        """thread safe"""
        cls = LargeInputExampleKernel
        fid = cls.binary_search_fid(item)
        examples = None
        for i, (f, exp) in enumerate(cls.cached_samples):
            if fid == f:
                examples = exp
                break
        if examples is None:
            print('------', item, fid, cls.indexes[:10], [f for f, _ in cls.cached_samples])
            print('++++++', [cls.stats[(sub, file)] for sub, file in cls.file_list[:10]],
                  hex(id(cls.cached_samples)), hex(id(cls)))
        assert examples is not None
        accu_count = cls.indexes[fid]
        assert item >= accu_count
        return examples[item - accu_count]
