from app.dataset import Dataset
from app.functions import TokenizeQuery, SplitValue
from app.valid_dataset import ValidDataset
from app import charactor_bert

types = {
    "SyntheticDataset": Dataset,
    "ValidDataset": ValidDataset,
    "TokenizeQuery": TokenizeQuery,
    "SplitValue": SplitValue,
    "character_bert.EncodeQuery": charactor_bert.EncodeQuery,
    "character_bert.Extractor": charactor_bert.Extractor,
}
