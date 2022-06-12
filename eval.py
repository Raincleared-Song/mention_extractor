import os
import argparse
import jsonlines
from tqdm import tqdm
from utils import FewNERDMetrics


def find_max_f1_txt(idx_file_list: list, valid_path: str, metric='f1'):
    max_idx, max_val, max_f1, max_pre, max_rec = -1, 0.0, 0.0, 0.0, 0.0
    for idx, f in idx_file_list:
        fin = open(os.path.join(valid_path, f), encoding='utf-8')
        cur_f1, cur_pre, cur_rec = -1., -1., -1.
        for line in fin.readlines():
            if line.startswith('micro_f1'):
                cur_f1 = float(line.strip().split(' ')[2])
            if line.startswith('precision'):
                cur_pre = float(line.strip().split(' ')[2])
            if line.startswith('recall'):
                cur_rec = float(line.strip().split(' ')[2])
        assert cur_f1 >= 0 and cur_pre >= 0 and cur_rec >= 0
        to_sub = cur_f1 if metric == 'f1' else (cur_pre if metric == 'pre' else cur_rec)
        if to_sub > max_val:
            max_idx = idx
            max_val = to_sub
            max_f1 = cur_f1
            max_pre = cur_pre
            max_rec = cur_rec
    return max_idx, max_f1, max_pre, max_rec


def find_max_f1_jsonl(idx_file_list: list, valid_path: str, metric='f1'):
    assert metric in ['f1', 'precision', 'recall']
    if metric == 'f1':
        metric = 'micro_f1'
    max_idx, max_val, max_f1, max_pre, max_rec = -1, 0.0, 0.0, 0.0, 0.0
    for idx, f in idx_file_list:
        evaluator = FewNERDMetrics()
        file_p = os.path.join(valid_path, f)
        reader = jsonlines.open(file_p)
        it_reader = reader if os.path.getsize(file_p) < 1e8 else tqdm(reader, desc=f)
        for line in it_reader:
            assert len(line['predict']) == len(line['label'])
            evaluator.expand([line['predict']], [line['label']])
        reader.close()
        results = evaluator.get_metrics()
        to_sub = results[metric]
        if to_sub > max_val:
            max_idx = idx
            max_val = to_sub
            max_f1 = results['micro_f1']
            max_pre = results['precision']
            max_rec = results['recall']
        pref = f[:-6].split('-')[1]
        valid_result_p = os.path.join(valid_path, f'valid-{pref}-{idx}.txt')
        if not os.path.exists(valid_result_p):
            fout = open(valid_result_p, 'w', encoding='utf-8')
            for key in sorted(results.keys()):
                fout.write("%s = %s\n" % (key, str(results[key])))
            fout.close()
    return max_idx, max_f1, max_pre, max_rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    parser.add_argument('--metric', '-m', help='metric to choose best model', default='f1',
                        choices=['f1', 'pre', 'rec'], type=str)
    parser.add_argument('--remove', '-rm', action='store_true', help='remove not best models')
    parser.add_argument('--jsonl', '-jl', action='store_true', help='find result by jsonline files')
    args = parser.parse_args()

    valid_path = os.path.join('checkpoint', args.task, 'valid')
    epoch_files = [(int(f[:-4].split('-')[2]), f) for f in os.listdir(valid_path) if f.startswith('valid-epoch')]
    epoch_files.sort()
    step_files = [(int(f[:-4].split('-')[2]), f) for f in os.listdir(valid_path) if f.startswith('valid-step')]
    step_files.sort()
    epoch_files_jl = [(int(f[:-6].split('-')[2]), f) for f in os.listdir(valid_path) if f.startswith('result-epoch')]
    epoch_files_jl.sort()
    step_files_jl = [(int(f[:-6].split('-')[2]), f) for f in os.listdir(valid_path) if f.startswith('result-step')]
    step_files_jl.sort()
    if args.jsonl:
        max_epoch, max_epoch_f1, max_epoch_pre, max_epoch_rec = find_max_f1_jsonl(
            epoch_files_jl, valid_path, args.metric)
    else:
        max_epoch, max_epoch_f1, max_epoch_pre, max_epoch_rec = find_max_f1_txt(epoch_files, valid_path, args.metric)
    print('max_epoch:', max_epoch)
    print('max_epoch_f1:', max_epoch_f1)
    print('max_epoch_pre:', max_epoch_pre)
    print('max_epoch_rec:', max_epoch_rec)
    if args.jsonl:
        max_step, max_step_f1, max_step_pre, max_step_rec = find_max_f1_jsonl(
            step_files_jl, valid_path, args.metric)
    else:
        max_step, max_step_f1, max_step_pre, max_step_rec = find_max_f1_txt(step_files, valid_path, args.metric)
    print('max_step:', max_step)
    print('max_step_f1:', max_step_f1)
    print('max_step_pre:', max_step_pre)
    print('max_step_rec:', max_step_rec)
    if args.remove:
        os.system(f'mv checkpoint/{args.task}/model/epoch-{max_epoch}.pkl checkpoint/{args.task}/')
        os.system(f'mv checkpoint/{args.task}/model/epoch-{epoch_files[-1][0]}.pkl checkpoint/{args.task}/')
        os.system(f'mv checkpoint/{args.task}/model/step-{max_step}.pkl checkpoint/{args.task}/')
        os.system(f'mv checkpoint/{args.task}/model/step-{step_files[-1][0]}.pkl checkpoint/{args.task}/')
        os.system(f'rm -rf checkpoint/{args.task}/model/*')
        os.system(f'mv checkpoint/{args.task}/*.pkl checkpoint/{args.task}/model/')


if __name__ == '__main__':
    main()
