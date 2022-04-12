import os
import time
import torch
import random
import argparse
import numpy as np
from config import Config
from models import name_to_model
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from preprocess import FewNERDBertCrfFormatter

name_to_optimizer = {
    'adam': Adam,
    'adamw': AdamW
}


def init_all(seed=None):
    args = init_args()
    save_config(args)
    init_seed(seed)
    Config.cur_mode = args.mode
    datasets = init_data(args)
    models = init_model(args)
    return args.mode, datasets, models


def init_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', '-m', help='train/test',
                            type=str, choices=['train', 'test'], required=True)
    arg_parser.add_argument('--checkpoint', '-c', help='path of the checkpoint file', default=None)
    return arg_parser.parse_args()


def save_config(args):
    if os.path.exists('config'):
        config_list = [os.path.join('config', f) for f in os.listdir('config')]
    else:
        config_list = ['config.py']
    cur_config = Config
    time_str = '-'.join(time.asctime(time.localtime(time.time())).split(' '))
    base_path = os.path.join(cur_config.output_path, cur_config.model_path)
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, f'config_bak_{time_str}_{args.mode}.py')
    fout = open(save_path, 'w', encoding='utf-8')
    if args.checkpoint is not None:
        fout.write(f'# checkpoint: {args.pretrain_bert}\n\n')
    for f_name in config_list:
        if f_name.endswith('.py'):
            fout.write('# ------' + f_name + '------\n')
            fin = open(f_name, 'r')
            fout.write(fin.read())
            fout.write('\n')
            fin.close()
    fout.close()


global_loader_generator = torch.Generator()


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_seed(seed):
    global global_loader_generator
    if seed is None:
        seed = Config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    global_loader_generator.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    determine = torch.use_deterministic_algorithms if 'use_deterministic_algorithms' in dir(torch) \
        else torch.set_deterministic
    determine(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def init_dataset(mode: str):
    if Config.model_name in ['BERT-Crf', 'BERT-BiLSTM-Crf', 'Bert-Token-Classification']:
        form = FewNERDBertCrfFormatter()
        batch_size = Config.per_gpu_batch_size[mode] * max(1, Config.n_gpu)
        shuffle = (mode != 'test')

        dataset = form.read(mode)
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=Config.reader_num,
            collate_fn=form.process, drop_last=(mode == 'train'),
            worker_init_fn=seed_worker, generator=global_loader_generator,
        )
    else:
        raise NotImplementedError('Invalid Model Name!')

    return dataloader


def init_data(args):
    datasets = {'train': None, 'valid': None, 'test': None}
    if args.mode == 'train':
        datasets['train'] = init_dataset('train')
        datasets['valid'] = init_dataset('valid')
    else:
        datasets['test'] = init_dataset('test')
    return datasets


def init_model(args):
    model = name_to_model[Config.model_name]()
    trained_epoch, global_step = -1, 0
    if 'cuda' in Config.main_device:
        model = model.to(Config.main_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
        os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    # multi-gpu training
    if Config.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(Config.n_gpu)))
        print('Parallel Training!')

    optimizer = name_to_optimizer[Config.optimizer](
        model.parameters(), lr=Config.learning_rate, eps=Config.adam_epsilon, weight_decay=Config.weight_decay
    )

    if args.checkpoint is None:
        if args.mode == 'test':
            raise RuntimeError('Test mode need a trained model!')
    else:
        params = torch.load(args.checkpoint)
        if hasattr(model, 'module'):
            model.module.load_state_dict(params['model'])
        else:
            model.load_state_dict(params['model'])
        if args.mode == 'train':
            trained_epoch = params['trained_epoch']
            if Config.optimizer == params['optimizer_name']:
                optimizer.load_state_dict(params['optimizer'])
            if 'global_step' in params:
                global_step = params['global_step']

    return {
        'model': model,
        'optimizer': optimizer,
        'trained_epoch': trained_epoch,
        'global_step': global_step
    }
