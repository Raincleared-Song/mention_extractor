from kernel import init_all, train, test


if __name__ == '__main__':
    mode, datasets, models = init_all()
    if mode == 'train':
        global_step, ave_loss, best_steps, best_epoch, best_step_f1, best_epoch_f1 = train(datasets, models)
        print(f'global step: {global_step}')
        print(f'average loss: {ave_loss}')
        print(f'best step: {best_steps}')
        print(f'best step result: {best_step_f1}')
        print(f'best epoch: {best_epoch}')
        print(f'best epoch result: {best_epoch_f1}')
    else:
        results = test(datasets, models['model'], 'test')
        print(f'test results: {results}')
