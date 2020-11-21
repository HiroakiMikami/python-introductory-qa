import numpy as np
import torch.utils.data as data
from app.generator import Generator
from mlprogram import Environment


class Dataset(data.IterableDataset):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        base_seed = self.rng.randint(0, 2 << 32 - 1)
        worker_info = data.get_worker_info()
        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id

        seed = base_seed + worker_id

        generator = Generator(np.random.RandomState(seed))
        while True:
            example = generator.create()
            yield Environment(inputs={"text_query": example.text},
                              supervisions={"ground_truth": example.code})
