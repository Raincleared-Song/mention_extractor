# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import os
import copy
import torch
import jsonlines
from tqdm import tqdm
from typing import List, Type, Tuple
from configs import ConfigBase
from .fewnerd_datasets import InputExample, InputFeatures, InputExampleDataset, LargeInputExampleDataset, \
    label_convert, read_examples_from_file


class FewNERDBertCrfFormatter:
    def __init__(self, task: str, config: Type[ConfigBase]):
        self.data = []
        self.task = task
        self.config = config

    def read(self, mode: str = None):
        """
        read cache data, if not exist, do process
        """
        if self.task == 'supervised':
            assert mode is not None
            if isinstance(self.config.data_path, str):
                # is pretrain
                return LargeInputExampleDataset(self.config, mode)
            data_dir = self.config.data_path[mode]
            prefix, _ = os.path.splitext(data_dir)
            model_to_cache = {
                'bert-base-uncased':   'bert',
                'bert-large-uncased':  'bert',
                'facebook/bart-base':  'bart',
                'facebook/bart-large': 'bart',
                't5-base':             't5',
                't5-large':            't5',
            }
            cache_path = f'{prefix}_{self.config.label_type}_cache_{model_to_cache[self.config.bert_path]}.pth'
            print(cache_path)
            if os.path.exists(cache_path) and not self.config.overwrite_cache:
                print('loading examples:', mode, self.config.label_type, '......')
                self.data = torch.load(cache_path)
            else:
                print('processing examples:', mode, self.config.label_type, '......')
                self.data = read_examples_from_file(data_dir, mode, self.config)
                torch.save(self.data, cache_path)
            return InputExampleDataset(copy.deepcopy(self.data))
        elif self.task == 'fewshot':
            data_dir = self.config.data_path
            prefix, _ = os.path.splitext(data_dir)
            cache_path = f'{prefix}_{self.config.label_type}_cache.pth'
            if os.path.exists(cache_path) and not self.config.overwrite_cache:
                print('loading examples', prefix, self.config.label_type, '......')
                self.data = torch.load(cache_path)
            else:
                print('processing examples', prefix, self.config.label_type, '......')
                self.data = read_fewshot_examples_from_jsonl(data_dir, self.config)
                torch.save(self.data, cache_path)
            dataset_pairs = []
            for support, query in self.data:
                dataset_pairs.append((InputExampleDataset(support), InputExampleDataset(query)))
            return dataset_pairs
        else:
            raise NotImplementedError('Invalid Task Name!')

    def process(self, batch: List[InputExample]):
        if isinstance(self.config.data_path, str) and self.config.task == 'supervised':
            if all(example.guid == 'SKIP' for example in batch):
                return {}
        model_to_tokens = {
            'bert-base-uncased':   ('[CLS]', '[SEP]'),
            'bert-large-uncased':  ('[CLS]', '[SEP]'),
            'facebook/bart-base':  ('<s>', '</s>'),
            'facebook/bart-large': ('<s>', '</s>'),
            't5-base':             ('<s>', '</s>'),
            't5-large':            ('<s>', '</s>'),
        }
        cls_token, sep_token = model_to_tokens[self.config.bert_path]
        max_word_len = max(sum(len(word) for word in example.proc_words) for example in batch)
        padding_len = min(self.config.max_seq_length, max_word_len + 2)
        features = convert_examples_to_features(batch, self.config.label2id, padding_len, self.config.tokenizer,
                                                cls_token_at_end=False,
                                                cls_token=cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=sep_token,
                                                # roberta uses an extra separator b/w pairs of sentences
                                                sep_token_extra=False,
                                                pad_on_left=False,
                                                pad_token_id=self.config.tokenizer.pad_token_id,
                                                pad_token_segment_id=0,
                                                pad_token_label_id=-100,
                                                )
        input_ids, token_type_ids, attention_masks, labels, flags, lengths = [], [], [], [], [], []
        guids, extra_labels, words = [], [], []
        for fea in features:
            input_ids.append(fea.input_ids)
            token_type_ids.append(fea.segment_ids)
            attention_masks.append(fea.input_mask)
            labels.append(fea.label_ids)
            flags.append(fea.flags)
            lengths.append(fea.length)
            for i in range(fea.length):
                assert labels[-1][i] != -100
            guids.append(fea.guid)
            extra_labels.append(fea.extra_labels)
            words.append(fea.words)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "token_type_ids": torch.LongTensor(token_type_ids),
            "attention_mask": torch.FloatTensor(attention_masks),
            "labels": torch.LongTensor(labels),
            "flags": torch.LongTensor(flags),
            "lengths": torch.LongTensor(lengths),
            "guids": guids,
            "words": words,
            "extra_labels": extra_labels,
        }


def read_fewshot_examples_from_jsonl(data_dir: str, config: Type[ConfigBase]) -> \
        List[Tuple[List[InputExample], List[InputExample]]]:
    prefix, _ = os.path.splitext(data_dir)
    res_pair_batch = []
    reader = jsonlines.open(data_dir)
    item_id = 0
    for item in tqdm(reader, desc=prefix):
        support, query = item['support'], item['query']
        assert len(support['word']) == len(support['label']) and len(query['word']) == len(query['label'])
        support_batch, query_batch = [], []
        for sid, (s_words, s_labels) in enumerate(zip(support['word'], support['label'])):
            assert len(s_words) == len(s_labels)
            cur_words, cur_labels, cur_proc_words = [], [], []
            last_lab = ''
            for word, lab in zip(s_words, s_labels):
                cur_words.append(word)
                cur_labels.append(label_convert(lab, lab != last_lab, config))
                cur_proc_words.append(config.tokenizer.tokenize(word) if len(word) > 0 else [])
                last_lab = lab
            support_batch.append(InputExample(guid=f"{prefix}-support-{item_id}-{sid}",
                                              words=cur_words, labels=cur_labels, proc_words=cur_proc_words))
        for qid, (q_words, q_labels) in enumerate(zip(query['word'], query['label'])):
            assert len(q_words) == len(q_labels)
            cur_words, cur_labels, cur_proc_words = [], [], []
            last_lab = ''
            for word, lab in zip(q_words, q_labels):
                cur_words.append(word)
                cur_labels.append(label_convert(lab, lab != last_lab, config))
                cur_proc_words.append(config.tokenizer.tokenize(word) if len(word) > 0 else [])
                last_lab = lab
            query_batch.append(InputExample(guid=f"{prefix}-query-{item_id}-{qid}",
                                            words=cur_words, labels=cur_labels, proc_words=cur_proc_words))
        res_pair_batch.append((support_batch, query_batch))
        item_id += 1
    reader.close()
    return res_pair_batch


def convert_examples_to_features(examples: List[InputExample],
                                 label2id: dict,
                                 max_seq_length: int,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token_id=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-100,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for ex_index, example in enumerate(examples):

        tokens, label_ids, flags = [], [], []
        for word_tokens, label in zip(example.proc_words, example.labels):
            assert len(word_tokens) > 0
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and 'X' for the remaining tokens
            label_ids.extend([label2id[label]] + [label2id['X']] * (len(word_tokens) - 1))
            flags.extend([1] + [0] * (len(word_tokens) - 1))
        assert sum(flags) == len(example.words) == len(example.labels) == len(example.proc_words)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            flags = flags[:(max_seq_length - special_tokens_count)]

        remain_words = len(example.words) - sum(flags)
        assert 0 <= remain_words <= len(example.labels)
        extra_labels = example.labels[-remain_words:] if remain_words > 0 else []

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label2id['[SEP]']]
        flags += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [label2id['[SEP]']]
            flags += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label2id['[CLS]']]
            flags += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label2id['[CLS]']] + label_ids
            flags = [0] + flags
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        token_len = len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token_id] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            flags = ([0] * padding_length) + flags
        else:
            input_ids += ([pad_token_id] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)
            flags += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(flags) == max_seq_length
        assert len(label_ids) == max_seq_length, print(len(label_ids), max_seq_length)

        features.append(
                InputFeatures(guid=example.guid,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              flags=flags,
                              length=token_len,
                              words=example.words,
                              extra_labels=extra_labels))
    return features
