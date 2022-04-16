import os
import argparse


def find_max_f1(idx_file_list: list, valid_path: str, metric='f1'):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='model name', type=str)
    parser.add_argument('--metric', '-m', help='metric to choose best model', choices=['f1', 'pre', 'rec'], type=str)
    args = parser.parse_args()

    valid_path = os.path.join('checkpoint', args.task, 'valid')
    epoch_files = [(int(f[:-4].split('-')[2]), f) for f in os.listdir(valid_path) if f.startswith('valid-epoch')]
    epoch_files.sort()
    step_files = [(int(f[:-4].split('-')[2]), f) for f in os.listdir(valid_path) if f.startswith('valid-step')]
    step_files.sort()
    max_epoch, max_epoch_f1, max_epoch_pre, max_epoch_rec = find_max_f1(epoch_files, valid_path, args.metric)
    print('max_epoch:', max_epoch)
    print('max_epoch_f1:', max_epoch_f1)
    print('max_epoch_pre:', max_epoch_pre)
    print('max_epoch_rec:', max_epoch_rec)
    max_step, max_step_f1, max_step_pre, max_step_rec = find_max_f1(step_files, valid_path, args.metric)
    print('max_step:', max_step)
    print('max_step_f1:', max_step_f1)
    print('max_step_pre:', max_step_pre)
    print('max_step_rec:', max_step_rec)


if __name__ == '__main__':
    main()
