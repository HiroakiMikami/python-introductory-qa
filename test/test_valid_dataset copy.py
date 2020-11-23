from app.valid_dataset import ValidDataset


def test_load():
    dataset = ValidDataset()
    assert len(dataset) == 97
