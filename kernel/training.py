import os
import torch
from .testing import test
from configs import ConfigBase
from utils import save_model
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def train(datasets, models, config: ConfigBase):
    train_dataset = datasets['train']
    train_sz = len(train_dataset)

    # 创建输出文件夹
    os.makedirs(config.output_path, exist_ok=True)
    task_path = os.path.join(config.output_path, config.model_path)
    os.makedirs(task_path, exist_ok=True)
    model_output_path = os.path.join(task_path, 'model')
    os.makedirs(model_output_path, exist_ok=True)
    valid_output_path = os.path.join(task_path, 'valid')
    os.makedirs(valid_output_path, exist_ok=True)

    if config.max_step > 0:
        total_t = config.max_step
        config.num_epoch = config.max_step // (train_sz // config.grad_accu_step) + 1
    else:
        total_t = train_sz // config.grad_accu_step * config.num_epoch

    model = models['model']
    optimizer = models['optimizer']
    trained_epoch = models['trained_epoch'] + 1
    global_step = models['global_step']
    train_batch_sz = config.per_gpu_batch_size['train']

    scheduler = None
    if config.num_warmup_steps >= 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=config.num_warmup_steps, num_training_steps=total_t
        )
    determine = torch.use_deterministic_algorithms if 'use_deterministic_algorithms' in dir(torch) \
        else torch.set_deterministic

    train_loss = 0.0
    best_step_f1, best_steps = -1, -1
    best_epoch_f1, best_epoch = -1, -1
    best_results = {}
    model.zero_grad()

    for epoch in range(trained_epoch, int(config.num_epoch)):
        print(f'training epoch {epoch} ......')
        epoch_iterator = tqdm(train_dataset, desc="data iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            del batch['guids'], batch['words'], batch['extra_labels']
            if config.n_gpu > 1:
                batch['rank'] = torch.LongTensor(list(range(config.n_gpu)))
            # transfer data to gpu
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(config.main_device, non_blocking=True)

            if config.model_name in ['BERT-Crf', 'BERT-BiLSTM-Crf', 'Bert-Token-Classification']:
                loss = model(**batch)
            else:
                raise NotImplementedError()

            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if config.grad_accu_step > 1:
                loss /= config.grad_accu_step

            determine(False)
            loss.backward()
            determine(True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            train_loss += loss.item()
            if (step + 1) % config.grad_accu_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                # 梯度累加
                optimizer.step()
                if config.num_warmup_steps >= 0:
                    scheduler.step()
                model.zero_grad()
                global_step += 1

                if config.save_step > 0 and global_step % config.save_step == 0:
                    results = test(datasets, model, 'valid', config, valid_output_path, step=global_step)
                    if results['eval_f1'] > best_step_f1:
                        best_step_f1 = results['eval_f1']
                        best_steps = global_step
                    if config.save_model:
                        # Save model checkpoint
                        cp_output_dir = os.path.join(model_output_path, f'step-{global_step}.pkl')
                        model_to_save = model.module if hasattr(model, "module") else model
                        save_model(cp_output_dir, model_to_save, config.optimizer, optimizer, epoch, global_step)

            if 0 < config.max_step < global_step:
                epoch_iterator.close()
                break
            if isinstance(config.data_path, str):
                # is pretrain
                next_begin, next_end = (step + 1) % train_sz, (step + 2) % train_sz
                train_dataset.dataset.kernel.check_status(next_begin * train_batch_sz, next_end * train_batch_sz)
        if 0 < config.max_step < global_step:
            break
        # save model for every epoch
        if epoch % config.save_epoch == 0:
            results = test(datasets, model, 'valid', config, valid_output_path, epoch=epoch)
            if results['eval_f1'] > best_epoch_f1:
                best_epoch_f1 = results['eval_f1']
                best_epoch = epoch
                best_results = results
            if config.save_model:
                # Save model checkpoint
                cp_output_dir = os.path.join(model_output_path, f'epoch-{epoch}.pkl')
                model_to_save = model.module if hasattr(model, "module") else model
                save_model(cp_output_dir, model_to_save, config.optimizer, optimizer, epoch, global_step)
    return global_step, train_loss / global_step, best_steps, best_epoch, best_step_f1, best_epoch_f1, best_results
