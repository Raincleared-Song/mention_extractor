import os
import torch
import jsonlines
from tqdm import tqdm
from config import Config
from utils import FewNERDMetrics, ParallelCollector
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calc_metrics(pred, labels, class_num):
    positive_labels = list(range(1, class_num))
    precision = precision_score(labels, pred, labels=positive_labels, average='micro')
    recall = recall_score(labels, pred, labels=positive_labels, average='micro')
    f1 = f1_score(labels, pred, labels=positive_labels, average='micro')
    accuracy = accuracy_score(labels, pred)
    return precision, recall, f1, accuracy


def test(datasets, model, mode: str, output_path: str = None, epoch=-1, step=-1):
    if Config.model_name in ['BERT-Crf', 'BERT-BiLSTM-Crf', 'Bert-Token-Classification']:
        return test_crf(datasets, model, mode, output_path, epoch, step)
    else:
        raise RuntimeError('invalid model_name')


def test_crf(datasets, model, mode: str, output_path: str = None, epoch=-1, step=-1):
    assert mode in ['valid', 'test']
    dataset = datasets[mode]
    eval_loss, eval_step_b = 0.0, 0

    evaluator = FewNERDMetrics()
    crf_pred, results = {}, {}

    assert mode == 'test' or output_path is not None and (epoch >= 0 or step >= 0)
    if mode == 'test':
        task_path = os.path.join(Config.output_path, Config.model_path)
        output_path = os.path.join(task_path, 'test')
        os.makedirs(output_path, exist_ok=True)
        output_file_name = os.path.join(output_path, 'result.jsonl')
    elif epoch != -1:
        output_file_name = os.path.join(output_path, f'result-epoch-{epoch}.jsonl')
    else:
        output_file_name = os.path.join(output_path, f'result-step-{step}.jsonl')
    writer = jsonlines.open(output_file_name, 'w')

    for batch in tqdm(dataset, desc=mode):
        model.eval()
        guids, words, extra_labels = batch['guids'], batch['words'], batch['extra_labels']
        del batch['guids'], batch['words'], batch['extra_labels']
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(Config.main_device)
        if Config.n_gpu > 1:
            batch['rank'] = torch.LongTensor(list(range(Config.n_gpu)))

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
    eval_loss = eval_loss / eval_step_b
    writer.close()

    results.update(evaluator.get_metrics())
    results['eval_loss'] = eval_loss
    if epoch == -1:
        file_name = f'{mode}-step-{step}.txt'
    else:
        file_name = f'{mode}-epoch-{epoch}.txt'
    output_file = os.path.join(output_path, file_name)
    fout = open(output_file, 'w')
    fout.write("model            = %s\n" % str(Config.model_path))
    fout.write("total batch size = %d\n" % (Config.per_gpu_batch_size['train'] * Config.grad_accu_step))
    fout.write("train num epochs = %d\n" % Config.num_epoch)
    fout.write("max seq length   = %d\n" % Config.max_seq_length)
    for key in sorted(results.keys()):
        fout.write("%s = %s\n" % (key, str(results[key])))
    fout.close()
    results['eval_f1'] = results['micro_f1']

    return results
