import os
from kernel import init_all, init_args, init_model, init_data, train, test


def main_supervised(args):
    datasets, models, config = init_all(args)
    if args.mode == 'train':
        global_step, ave_loss, best_steps, best_epoch, best_step_f1, best_epoch_f1, best_results = \
            train(datasets, models, config)
        output_file = os.path.join(config.output_path, config.model_path, f'results-{args.mode}.txt')
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write(f'global step: {global_step}\n')
            fout.write(f'average loss: {ave_loss}\n')
            fout.write(f'best step: {best_steps}\n')
            fout.write(f'best step result: {best_step_f1}\n')
            fout.write(f'best epoch: {best_epoch}\n')
            fout.write(f'best epoch result: {best_epoch_f1}\n')
        if args.do_test:
            if best_step_f1 > best_epoch_f1:
                model_name = f'step-{best_steps}.pkl'
            else:
                model_name = f'epoch-{best_epoch}.pkl'
            args.mode = 'test'
            args.checkpoint = f'checkpoint/{config.model_path}/model/{model_name}'
            del datasets, models
            datasets = init_data(args)
            models = init_model(args)
    if args.mode == 'test':
        results = test(datasets, models['model'], 'test', config)
        output_file = os.path.join(config.output_path, config.model_path, f'results-{args.mode}.txt')
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write(f'test results: {results}\n')


def main_fewshot(args):
    datasets, models, config = init_all(args)
    model_d, optimizer_d = models['model'].state_dict(), models['optimizer'].state_dict()
    total_predict, total_instance, total_correct = 0, 0, 0
    for (did, dataset) in enumerate(datasets):
        print('------', 'pair index:', did, '------')
        _, _, _, _, _, _, results = train(dataset, models, config)
        models['model'].load_state_dict(model_d)
        models['optimizer'].load_state_dict(optimizer_d)
        total_correct += results['correct_cnt']
        total_instance += results['instance_cnt']
        total_predict += results['predict_cnt']
    print(total_predict, total_instance, total_correct)
    output_file = os.path.join(config.output_path, config.model_path, 'results-fewshot.txt')
    fout = open(output_file, 'w', encoding='utf-8')
    print('total_correct:', total_correct, file=fout)
    print('total_predict:', total_predict, file=fout)
    print('total_instance:', total_instance, file=fout)
    pre = total_correct / (total_predict + 1e8)
    rec = total_correct / (total_instance + 1e8)
    f1 = 2 * pre * rec / (pre + rec + 1e8)
    print('precision:', round(pre * 100, 2), file=fout)
    print('recall:', round(rec * 100, 2), file=fout)
    print('f1:', round(f1 * 100, 2), file=fout)
    fout.close()


if __name__ == '__main__':
    main_args = init_args()
    if main_args.task == 'supervised':
        main_supervised(main_args)
    elif main_args.task == 'fewshot':
        main_fewshot(main_args)
    else:
        raise NotImplementedError('Invalid Task Name!')
