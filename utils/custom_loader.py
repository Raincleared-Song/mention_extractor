import threading


class CustomDataloader:
    """
    self-implemented dataloader with no
    shuffle = False, num_workers = batch_size,
    """
    def __init__(self, dataset, batch_size: int, drop_last, num_workers=1, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.data_length = len(dataset)
        self.length = self.data_length // batch_size
        if not drop_last:
            self.length += int(self.length % batch_size > 0)

        self.ptr = 0
        self.thread_ptr = 0
        self.thread_queue = []

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def clear_pool(self):
        self.thread_ptr -= len(self.thread_queue)
        self.thread_queue.clear()

    def __next__(self):
        if self.ptr == self.length:
            assert self.ptr == self.thread_ptr == self.length
            assert len(self.thread_queue) == 0
            self.ptr = self.thread_ptr = 0
            raise StopIteration
        while self.thread_ptr < self.length and len(self.thread_queue) < self.num_workers:
            thr = CustomThread(self.thread_ptr, self)
            thr.start()
            self.thread_queue.append(thr)
            self.thread_ptr += 1
        assert len(self.thread_queue) > 0
        pop_thr: CustomThread = self.thread_queue[0]
        assert pop_thr.item == self.ptr
        pop_thr.join()
        try:
            assert pop_thr.result != [], 'CustomThread Execution Error!'
        except AssertionError:
            from IPython import embed
            embed()
        self.thread_queue = self.thread_queue[1:]
        self.ptr += 1
        return pop_thr.result


class CustomThread(threading.Thread):
    def __init__(self, item: int, loader: CustomDataloader):
        super().__init__()
        self.loader = loader
        self.item = item
        self.result = []

    def run(self):
        loader = self.loader
        begin, end = loader.batch_size * self.item, min(loader.batch_size * (self.item + 1), loader.data_length)
        assert begin < end
        batch = []
        for k in range(begin, end):
            batch.append(loader.dataset[k])
        if loader.collate_fn is None:
            self.result = batch
        else:
            self.result = loader.collate_fn(batch)
