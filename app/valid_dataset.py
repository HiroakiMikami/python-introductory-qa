import torch
import os
import csv
from mlprogram import Environment


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self):
        path = os.path.dirname(__file__)
        path = os.path.join(path, "..", "data", "test_dataset.csv")
        samples = []
        with open(path) as file:
            reader = csv.reader(file, delimiter=',', quotechar='"')
            for row in reader:
                # test dataset,text_query,ground_truth,simplified,source
                id, text_query, ground_truth, simplified, original = row
                samples.append(Environment(
                    inputs={
                        "text_query": text_query,
                    },
                    supervisions={
                        "id": id,
                        "ground_truth": ground_truth,
                        "original": original
                    }
                ))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
