from config import Config


class ParallelCollector:
    label_container = [[] for _ in range(Config.n_gpu)]
    true_container = [[] for _ in range(Config.n_gpu)]
