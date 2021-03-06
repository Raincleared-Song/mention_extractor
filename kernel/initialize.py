import os
import time
import torch
import random
import argparse
import numpy as np
from utils import CustomDataloader
from configs import task_to_config
from models import name_to_model
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from preprocess import FewNERDBertCrfFormatter

name_to_optimizer = {
    'adam': Adam,
    'adamw': AdamW
}


def init_all(args, seed=None):
    init_seed(args, seed)
    datasets = init_data(args)
    models = init_model(args)
    return datasets, models, task_to_config[args.task]


def init_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task', '-t', help='supervised/fewshot',
                            type=str, choices=['supervised', 'fewshot'], required=True)
    arg_parser.add_argument('--mode', '-m', help='train/test',
                            type=str, choices=['train', 'test', 'both'], required=True)
    arg_parser.add_argument('--checkpoint', '-c', help='path of the checkpoint file', default=None)
    arg_parser.add_argument('--teacher_checkpoint', '-tc', help='path of the teacher model', default=None)
    arg_parser.add_argument('--part', help='fewshot data part', type=str, default='')
    arg_parser.add_argument('--n_way', help='fewshot number of ways', type=int, default=0)
    arg_parser.add_argument('--n_shot', help='fewshot number of shots', type=int, default=0)
    arg_parser.add_argument('--device', help='the device used', type=str, default='cuda:0')
    arg_parser.add_argument('--resume', help='if set, skip trained steps', action='store_true')
    args = arg_parser.parse_args()
    if args.mode == 'both':
        args.mode = 'train'
        args.do_test = True
    else:
        args.do_test = False
    save_config(args)
    return args


def save_config(args):
    cur_config = task_to_config[args.task]
    if args.task == 'fewshot':
        assert args.part != '' and args.n_way != 0 and args.n_shot != 0
        cur_config.part = args.part
        cur_config.n_way = args.n_way
        cur_config.n_shot = args.n_shot
        cur_config.data_path = f'data/episode-data/{args.part}/test_{args.n_way}_{args.n_shot}.jsonl'
        cur_config.model_path = cur_config.model_path.format(args.part, args.n_way, args.n_shot)
        cur_config.max_seq_length = cur_config.max_seq_length_map[(args.n_way, args.n_shot)]
    cur_config.main_device = args.device
    cur_config.skip_trained_steps = args.resume
    config_list = [os.path.join('configs', f) for f in os.listdir('configs')]
    time_str = '-'.join(time.asctime(time.localtime(time.time())).split(' '))
    base_path = os.path.join(cur_config.output_path, cur_config.model_path)
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, f'config_bak_{time_str}_{args.mode}.py')
    fout = open(save_path, 'w', encoding='utf-8')
    if args.checkpoint is not None:
        fout.write(f'# checkpoint: {args.checkpoint}\n\n')
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


def init_seed(args, seed):
    global global_loader_generator
    if seed is None:
        seed = task_to_config[args.task].seed
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
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def init_dataset(task: str, mode: str = None):
    config = task_to_config[task]
    if task == 'supervised':
        assert mode is not None
        form = FewNERDBertCrfFormatter(task, config)
        batch_size = config.per_gpu_batch_size[mode] * max(1, config.n_gpu)
        is_pretrain = isinstance(config.data_path, str) and config.task == 'supervised'
        dataset = form.read(mode)
        if is_pretrain:
            dataloader = CustomDataloader(dataset=dataset, batch_size=batch_size, drop_last=(mode == 'train'),
                                          num_workers=config.reader_num, collate_fn=form.process)
        else:
            dataloader = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=config.reader_num,
                collate_fn=form.process, drop_last=(mode == 'train'),
                worker_init_fn=seed_worker, generator=global_loader_generator, pin_memory=True,
            )
    elif task == 'fewshot':
        form = FewNERDBertCrfFormatter(task, config)
        datasets = form.read()
        print('got few shot pairs:', len(datasets))
        dataloader = fewshot_batch_to_loader(datasets, config, form)
    else:
        raise NotImplementedError('Invalid Task Name!')

    return dataloader


def fewshot_batch_to_loader(datasets, config, form: FewNERDBertCrfFormatter):
    for support, query in datasets:
        support_loader = DataLoader(
            dataset=support, batch_size=config.per_gpu_batch_size['train'], shuffle=True,
            collate_fn=form.process, drop_last=False,
            worker_init_fn=seed_worker, generator=global_loader_generator, pin_memory=True,
        )
        query_loader = DataLoader(
            dataset=query, batch_size=config.per_gpu_batch_size['valid'], shuffle=False,
            collate_fn=form.process, drop_last=False,
            worker_init_fn=seed_worker, generator=global_loader_generator, pin_memory=True,
        )
        yield {'train': support_loader, 'valid': query_loader}


def init_data(args):
    if args.task == 'supervised':
        datasets = {'train': None, 'valid': None, 'test': None}
        if args.mode == 'train':
            datasets['train'] = init_dataset(args.task, 'train')
            datasets['valid'] = init_dataset(args.task, 'valid')
        else:
            datasets['test'] = init_dataset(args.task, 'test')
    elif args.task == 'fewshot':
        datasets = init_dataset(args.task)
    else:
        raise NotImplementedError('Invalid Task Name!')
    return datasets


def init_model(args):
    config = task_to_config[args.task]
    model = name_to_model[config.model_name](config)
    trained_epoch, global_step = -1, 0
    scheduler = None
    if 'cuda' in config.main_device:
        torch.cuda.set_device(config.main_device)
        torch.cuda.empty_cache()
        model = model.to(config.main_device)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    # multi-gpu training
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(config.n_gpu)))
        print('Parallel Training!')

    optimizer = name_to_optimizer[config.optimizer](
        model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon, weight_decay=config.weight_decay
    )

    if args.checkpoint is None:
        if args.mode == 'test':
            raise RuntimeError('Test mode need a trained model!')
    else:
        params = torch.load(args.checkpoint, map_location={f'cuda:{k}': config.main_device for k in range(8)})
        print('Loaded student model from:', args.checkpoint, '......')
        if hasattr(model, 'module'):
            model.module.load_state_dict(params['model'])
        else:
            model.load_state_dict(params['model'])
        if args.mode == 'train':
            trained_epoch = params['trained_epoch']
            if config.optimizer == params['optimizer_name']:
                optimizer.load_state_dict(params['optimizer'])
            if 'global_step' in params:
                global_step = params['global_step']
        if 'scheduler' in params:
            scheduler = params['scheduler']
    teacher_model = None
    if config.self_training:
        if args.teacher_checkpoint is None:
            if args.mode != 'test':
                raise RuntimeError('Self-training needs a teacher model!')
        else:
            teacher_model = name_to_model[config.model_name](config)
            if 'cuda' in config.main_device:
                teacher_model = teacher_model.to(config.main_device)
            params = torch.load(args.teacher_checkpoint,
                                map_location={f'cuda:{k}': config.main_device for k in range(8)})
            teacher_model.load_state_dict(params['model'])
            print('Loaded teacher model from:', args.teacher_checkpoint, '......')
            if not config.re_initialize and args.checkpoint is None:
                # only when the checkpoint is None
                model.load_state_dict(teacher_model.state_dict())
                print('Initialize the student through the weights of teacher!')
            elif args.checkpoint is not None:
                print('Initialize the student with the checkpoint!')
            else:
                print('Re-initialize the student!')

    return {
        'model': model,
        'optimizer': optimizer,
        'trained_epoch': trained_epoch,
        'global_step': global_step,
        'scheduler': scheduler,
        'teacher': teacher_model,
    }
