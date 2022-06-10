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
                if len(cur_proc) == 0:
                    cur_proc = ['[unused99]']
                if len(cur_word) > 0:
                    assert len(cur_proc) > 0
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


class LargeInputExampleDataset(Dataset):
    """large dataset for pretraining"""
    def __init__(self, config: Type[ConfigBase], mode: str):
        print('Initializing large dataset ......')
        self.config = config
        assert isinstance(self.config.data_path, str)
        self.data_path = config.data_path
        files = load_json(os.path.join(self.data_path, 'splits.json'))[mode]
        files = [(sub, file) for sub, file in files]
        if mode == 'train':
            random.shuffle(files)
        self.file_list = files
        stats = load_json(os.path.join(self.data_path, 'stats.json'))
        stats = {(sub, file): cnt for sub, file, cnt in stats}
        # create indexes
        self.indexes = [0]
        cur_sum = 0
        for sub, file in files:
            cnt = stats[(sub, file)]
            # assert cnt > self.config.per_gpu_batch_size[mode]
            cur_sum += cnt
            self.indexes.append(cur_sum)
        self.stats = stats
        self.total_len = cur_sum
        print('loaded file number:', len(self.file_list))
        print('loaded sample number:', cur_sum)

        # if size larger than 5, remove the earliest one
        self.pool_limit = 5
        self.preload_count = 3
        self.cached_samples = []  # List[fid, examples]
        self.cur_progress = 0  # the next fid to load
        # preload files
        for _ in range(self.preload_count):
            self.push_back(to_pop=False)

    def push_back(self, to_pop=True):
        """thread not safe"""
        fid = self.cur_progress
        if fid in [f for f, _ in self.cached_samples]:
            return
        sub, file = self.file_list[fid]
        file_path = os.path.join(self.data_path, sub, file)
        examples = read_examples_from_file(file_path, f'{sub}+{file}', self.config)
        self.cached_samples.append((fid, examples))
        if len(examples) != self.stats[(sub, file)]:
            print('******', sub, file, '******')
        assert len(examples) == self.stats[(sub, file)]
        self.cur_progress = (fid + 1) % len(self.file_list)
        if to_pop and len(self.cached_samples) > self.pool_limit:
            self.pop()

    def pop(self):
        """thread not safe"""
        assert len(self.cached_samples) > 0
        self.cached_samples = self.cached_samples[1:]

    def check_status(self, next_begin: int, next_end: int, to_push=True):
        """thread not safe"""
        sid, eid = self.binary_search_fid(next_begin), self.binary_search_fid(next_end - 1)
        if sid != eid or next_begin == self.indexes[sid]:
            if to_push:
                self.push_back()
            return True
        return False

    def binary_search_fid(self, did):
        """thread safe"""
        assert did < self.total_len
        lo, hi = 0, len(self.indexes)
        while lo < hi:
            mi = (lo + hi) // 2
            if self.indexes[mi] <= did:
                lo = mi + 1
            else:
                hi = mi
        return lo - 1

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        """thread safe"""
        fid = self.binary_search_fid(item)
        examples = None
        for i, (f, exp) in enumerate(self.cached_samples):
            if fid == f:
                examples = exp
                break
        if examples is None:
            print('------', item, fid, self.indexes[:10], [f for f, _ in self.cached_samples])
            print('++++++', [self.stats[(sub, file)] for sub, file in self.file_list[:10]],
                  hex(id(self.cached_samples)), hex(id(self)))
        assert examples is not None
        accu_count = self.indexes[fid]
        assert item >= accu_count
        return examples[item - accu_count]
