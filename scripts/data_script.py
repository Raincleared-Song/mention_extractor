import os
import csv
import json
import random
import jsonlines
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def gen_labels_json():
    labels = ['O']
    labels_bio = ['O']
    fin = open('data/labels.txt', encoding='utf-8')
    for line in fin.readlines():
        line = line.strip()
        if len(line) == 0 or line == 'O':
            continue
        labels.append(line)
        labels_bio.append('B-' + line)
        labels_bio.append('I-' + line)
    print(len(labels))
    save_json(labels, 'data/labels.json')
    save_json(labels_bio, 'data/labels_bio.json')


def gen_fewnerd_temp_ner(part: str):
    # conll2003/train.csv: 27586 23222
    # [(1, 17850), (2, 8333), (3, 989), (4, 292), (5, 78), (6, 24), (7, 16), (10, 3), (8, 1)]
    from configs import ConfigSupervised as Config
    assert Config.label_type == 'type'
    """
    fin = open('../templateNER/data/conll2003/train.csv', encoding='utf-8')
    reader = csv.reader(fin)
    token_len, cnt_neg, cnt_pos = {}, 0, 0
    for line in reader:
        assert len(line) == 2
        pos = line[1].find(' is not a named entity')
        if pos != -1:
            ll = len(line[1][:pos].split(' '))
            token_len.setdefault(ll, 0)
            token_len[ll] += 1
            cnt_neg += 1
        else:
            cnt_pos += 1
    fin.close()
    print(cnt_neg, cnt_pos)
    token_len = sorted(list(token_len.items()), key=lambda x: x[1], reverse=True)
    print(token_len)
    total = sum(x[1] for x in token_len)
    print(sum(x[0] * x[1] / total for x in token_len))
    exit()
    """
    from preprocess.fewnerd_formatter import read_examples_from_file

    poisson_lambda = 2
    random.seed(100)
    np.random.seed(200)
    os.makedirs(f'data/{part}_template_mention', exist_ok=True)
    for p in ['train', 'dev', 'test']:
        examples = read_examples_from_file(f'data/{part}/{p}.txt', '', Config)
        fout = open(f'data/{part}_template_mention/{p}.csv', 'w', newline='', encoding='utf-8')
        writer = csv.writer(fout)
        total_len, total_cnt, max_len, min_len = 0, 0, 0, 10000
        for exp in tqdm(examples, desc=p):
            words, labels = exp.words, exp.labels
            max_len, min_len = max(max_len, len(words)), min(min_len, len(words))
            assert len(words) == len(labels)
            last, positive_num = 0, 0
            positive_spans = []
            while last < len(labels):
                if labels[last] != 'O':
                    start = last
                    cur_label = labels[last]
                    last += 1
                    while last < len(labels) and labels[last] == cur_label:
                        last += 1
                    writer.writerow([' '.join(words), ' '.join(words[start:last]) + ' is a named entity'])
                    positive_num += 1
                    total_len += last - start
                    positive_spans.append((start, last))
                else:
                    last += 1
            total_cnt += positive_num
            # for negative samples
            if p == 'test':
                continue
            negative_num = int(1.5 * positive_num)
            sampled_lengths = np.random.poisson(poisson_lambda, negative_num)
            length_to_count = {}
            for ll in sampled_lengths:
                length_to_count.setdefault(ll, 0)
                length_to_count[ll] += 1
            for ll, cnt in length_to_count.items():
                permit_start = set(range(0, len(words) - ll + 1))
                for start, end in positive_spans:
                    if end - start == ll:
                        permit_start.discard(start)
                if len(permit_start) == 0:
                    continue
                permit_start = list(permit_start)
                for _ in range(cnt):
                    start = random.choice(permit_start)
                    last = start + ll
                    writer.writerow([' '.join(words), ' '.join(words[start:last]) + ' is not a named entity'])
        fout.close()
        # train 680543 340387 1.9993213606865128 267 1
        # dev 680543 340387 1.9993213606865128 156 1
        # test 193135 96902 1.9930961177271882 299 1
        print(p, total_len, total_cnt, total_len / total_cnt, max_len, min_len)


def gen_fewnerd_entlm_ner(part: str):
    import torch
    label_frac = {"": {}}  # calculated by training set
    for p in ['train', 'dev', 'test']:
        cache_path = f'data/{part}/{p}_type_bio_cache.pth'
        assert os.path.exists(cache_path)
        data = torch.load(cache_path)

        writer = jsonlines.open(f'data/{part}/{p}_entlm.jsonl', 'w')
        for example in tqdm(data, desc=p):
            assert len(example.words) == len(example.labels) == len(example.proc_words)
            writer.write({'text': example.words, 'label': example.labels})
            if p != 'train':
                continue
            for proc_word, label in zip(example.proc_words, example.labels):
                if label == 'O':
                    continue
                assert label[:2] in ('B-', 'I-')
                label = label[2:]
                if label not in label_frac:
                    label_frac[label] = {}
                cur_dict = label_frac[label]
                for token in proc_word:
                    cur_dict.setdefault(token, 0)
                    cur_dict[token] += 1
        writer.close()
    save_json(label_frac, f'data/{part}/label_frac.json')


def convert_to_standard_bio(part: str):
    import torch
    for p in ['train', 'dev', 'test']:
        cache_path = f'data/{part}/{p}_type_bio_cache.pth'
        assert os.path.exists(cache_path)
        data = torch.load(cache_path)
        fout = open(f'../BARTNER/data/fewnerd/supervised/{p}.txt', 'w', encoding='utf-8')
        for example in tqdm(data, desc=p):
            assert len(example.words) == len(example.labels) == len(example.proc_words)
            for word, label in zip(example.words, example.labels):
                fout.write(f'{word}\t{label}\n')
            fout.write('\n')
        fout.close()


def convert_fewshot_to_standard_type(part: str, standard_bio=False):
    file_path = f'data/episode-data/{part}'
    file_list = [f for f in os.listdir(file_path) if f.endswith('.jsonl')]
    for f in tqdm(file_list, desc=part):
        reader = jsonlines.open(f'{file_path}/{f}')
        # instance_count = 0
        # for item in reader:
        #     sents, labs = item['query']['word'], item['query']['label']
        #     assert len(sents) == len(labs)
        #     for sent, lab in zip(sents, labs):
        #         assert len(sent) == len(lab)
        #         last_lab = ''
        #         for s, la in zip(sent, lab):
        #             if la != 'O' and la != last_lab:
        #                 instance_count += 1
        #             last_lab = la
        fout = open(f'{file_path}/{f[:-6]}{"-bio" if standard_bio else ""}.txt', 'w')
        for item in reader:
            sents = item['support']['word'] + item['query']['word']
            labs = item['support']['label'] + item['query']['label']
            assert len(sents) == len(labs)
            for sent, lab in zip(sents, labs):
                assert len(sent) == len(lab)
                last_lab = ''
                for s, la in zip(sent, lab):
                    if not standard_bio or la == 'O':
                        pre_la = la
                    elif la == last_lab:
                        pre_la = 'I-' + la
                    else:
                        pre_la = 'B-' + la
                    last_lab = la
                    fout.write(f'{s}\t{pre_la}\n')
                fout.write('\n')
        fout.close()
        reader.close()


def create_file_indexes(path: str):
    random.seed(100)
    split_d = {
        'train': [], 'valid': [],
    }
    results = []
    sub_folders = sorted([p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))])
    for sub in tqdm(sub_folders):
        sub_path = os.path.join(path, sub)
        for file in sorted(os.listdir(sub_path)):
            cnt, lid = 0, 0
            sub_file_path = os.path.join(sub_path, file)
            fin = open(sub_file_path, encoding='utf-8')
            cur_words = []
            for line in fin.readlines():
                line = line.strip()
                if line.startswith("-DOCSTART-") or not line.strip():
                    if len(cur_words) > 0:
                        cnt += 1
                        cur_words = []
                else:
                    splits = line.split("\t")
                    if len(splits) != 2:
                        assert len(splits) == 1
                        splits = ['[unused99]'] + splits
                    assert len(splits) == 2
                    cur_word = splits[0].strip()
                    cur_words.append(cur_word)
                lid += 1
            if cur_words:
                cnt += 1
            fin.close()
            results.append((sub, file, cnt))
            rand = random.random()
            if rand <= 0.9:
                split_d['train'].append((sub, file))
            else:
                split_d['valid'].append((sub, file))
    save_json(results, os.path.join(path, 'stats.json'))
    save_json(split_d, os.path.join(path, 'splits.json'))
    print(len(split_d['train']), len(split_d['valid']))  # 15029 1701


if __name__ == '__main__':
    # gen_fewnerd_temp_ner('supervised')
    # gen_fewnerd_entlm_ner('supervised')
    # gen_labels_json()
    # convert_to_standard_bio('supervised')
    # convert_fewshot_to_standard_type('inter')
    # convert_fewshot_to_standard_type('intra')
    # convert_fewshot_to_standard_type('inter', standard_bio=True)
    # convert_fewshot_to_standard_type('intra', standard_bio=True)
    create_file_indexes('../project-tencent/data/processed2_txt')
