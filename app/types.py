from app.dataset import Dataset
from app.functions import TokenizeQuery, SplitValue
from app.valid_dataset import ValidDataset

types = {
    "SyntheticDataset": Dataset,
    "ValidDataset": ValidDataset,
    "TokenizeQuery": TokenizeQuery,
    "SplitValue": SplitValue,
}
