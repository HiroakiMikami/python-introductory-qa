from app import charactor_bert
from app.valid_dataset import ValidDataset
from app.functions import TokenizeQuery, SplitValue
from app.dataset import Dataset
import torch
from torch import nn


def Adam(model: nn.Module, *args, **kwargs):
    return torch.optim.Optimizer(
        [
            {"params": model.decoder.parameters()},
            {"params": model.encoder.parameters(), "lr": kwargs["lr"] * 0.1},
        ],
        *args,
        **kwargs,
    )


types = {
    "SyntheticDataset": Dataset,
    "ValidDataset": ValidDataset,
    "TokenizeQuery": TokenizeQuery,
    "SplitValue": SplitValue,
    "character_bert.EncodeQuery": charactor_bert.EncodeQuery,
    "character_bert.Extractor": charactor_bert.Extractor,
    "Adam": Adam
}
