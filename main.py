import os
from config import Config
from kernel import init_all, init_model, init_data, train, test


if __name__ == '__main__':
    args, datasets, models = init_all()
    if args.mode == 'train':
        global_step, ave_loss, best_steps, best_epoch, best_step_f1, best_epoch_f1 = train(datasets, models)
        output_file = os.path.join(Config.output_path, Config.model_path, f'results-{args.mode}.txt')
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
            args.checkpoint = f'checkpoint/{Config.model_path}/model/{model_name}'
            del datasets, models
            datasets = init_data(args)
            models = init_model(args)
    if args.mode == 'test':
        results = test(datasets, models['model'], 'test')
        output_file = os.path.join(Config.output_path, Config.model_path, f'results-{args.mode}.txt')
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write(f'test results: {results}\n')
