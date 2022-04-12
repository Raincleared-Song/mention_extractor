import json


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


if __name__ == '__main__':
    gen_labels_json()
