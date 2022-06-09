import os
import torch
import jsonlines
from tqdm import tqdm
from configs import ConfigBase
from utils import FewNERDMetrics, ParallelCollector


def test(datasets, model, mode: str, config: ConfigBase, output_path: str = None, epoch=-1, step=-1):
    if config.model_name in ['BERT-Crf', 'BERT-BiLSTM-Crf', 'Bert-Token-Classification']:
        return test_crf(datasets, model, mode, config, output_path, epoch, step)
    else:
        raise RuntimeError('invalid model_name')


def test_crf(datasets, model, mode: str, config: ConfigBase, output_path: str = None, epoch=-1, step=-1):
    assert mode in ['valid', 'test']
    dataset = datasets[mode]
    eval_loss, eval_step_b = 0.0, 0
    test_sz, test_batch_sz = len(dataset), config.per_gpu_batch_size[mode]

    evaluator = FewNERDMetrics(config)
    crf_pred, results = {}, {}

    assert mode == 'test' or output_path is not None and (epoch >= 0 or step >= 0)
    if mode == 'test':
        task_path = os.path.join(config.output_path, config.model_path)
        output_path = os.path.join(task_path, 'test')
        os.makedirs(output_path, exist_ok=True)
        output_file_name = os.path.join(output_path, 'result.jsonl')
    elif epoch != -1:
        output_file_name = os.path.join(output_path, f'result-epoch-{epoch}.jsonl')
    else:
        output_file_name = os.path.join(output_path, f'result-step-{step}.jsonl')
    writer = jsonlines.open(output_file_name, 'w')

    for step, batch in enumerate(tqdm(dataset, desc=mode)):
        model.eval()
        guids, words, extra_labels = batch['guids'], batch['words'], batch['extra_labels']
        del batch['guids'], batch['words'], batch['extra_labels']
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(config.main_device, non_blocking=True)
        if config.n_gpu > 1:
            batch['rank'] = torch.LongTensor(list(range(config.n_gpu)))

        with torch.no_grad():
            tmp_eval_loss = model(**batch)

            predict_labels, true_labels = [], []
            for item in ParallelCollector.label_container:
                predict_labels += item
            for item in ParallelCollector.true_container:
                true_labels += item
            eval_loss += tmp_eval_loss.mean().item()
            # add extra labels
            assert len(guids) == len(words) == len(extra_labels) == len(predict_labels) == len(true_labels)
            for idx in range(len(guids)):
                predict_labels[idx] += ["O"] * len(extra_labels[idx])
                true_labels[idx] += extra_labels[idx]
                assert len(predict_labels[idx]) == len(true_labels[idx]) == len(words[idx])
                writer.write({
                    "guid": guids[idx],
                    "words": words[idx],
                    "predict": predict_labels[idx],
                    "label": true_labels[idx],
                })
            # handle omitted samples
            evaluator.expand(predict_labels, true_labels)
        eval_step_b += 1

        if isinstance(config.data_path, str):
            # is pretrain
            next_begin, next_end = (step + 1) % test_sz, (step + 2) % test_sz
            dataset.dataset.kernel.check_status(next_begin * test_batch_sz, next_end * test_batch_sz)

    eval_loss = eval_loss / eval_step_b
    writer.close()

    del tmp_eval_loss
    torch.cuda.empty_cache()

    results.update(evaluator.get_metrics())
    results['eval_loss'] = eval_loss
    if mode == 'test':
        file_name = f'{mode}-result.txt'
    elif epoch == -1:
        file_name = f'{mode}-step-{step}.txt'
    else:
        file_name = f'{mode}-epoch-{epoch}.txt'
    output_file = os.path.join(output_path, file_name)
    fout = open(output_file, 'w')
    fout.write("model            = %s\n" % str(config.model_path))
    fout.write("total batch size = %d\n" % (config.per_gpu_batch_size['train'] * config.grad_accu_step))
    fout.write("train num epochs = %d\n" % config.num_epoch)
    fout.write("max seq length   = %d\n" % config.max_seq_length)
    for key in sorted(results.keys()):
        fout.write("%s = %s\n" % (key, str(results[key])))
    fout.close()
    results['eval_f1'] = results['micro_f1']

    return results
