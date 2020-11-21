from app.dataset import Dataset
import torch
import ast


def test_generation():
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2,
                                         collate_fn=lambda batch: batch)
    b0 = None
    b1 = None
    for i, batch in enumerate(loader):
        if i == 0:
            b0 = batch
        else:
            b1 = batch
            break

    assert b0 != b1

    for i, batch in enumerate(loader):
        example = batch[0]
        ast.parse(example.supervisions["ground_truth"])
        if i == 10000:
            break
