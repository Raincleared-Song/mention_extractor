from configs import ConfigBase as Config


class ParallelCollector:
    label_container = [[] for _ in range(Config.n_gpu)]
    true_container = [[] for _ in range(Config.n_gpu)]
    predict_container = [[] for _ in range(Config.n_gpu)]
