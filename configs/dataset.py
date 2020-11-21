train_dataset_seed = 1234
test_dataset_seed = 4321
n_test_dataset = 100

is_subtype = mlprogram.languages.python.IsSubtype()
normalize_dataset = mlprogram.utils.transform.NormalizeGroundTruth(
    normalize=mlprogram.functools.Sequence(
        funcs=collections.OrderedDict(
            items=[["parse", parser.parse], ["unparse", parser.unparse]],
        ),
    ),
)

train_dataset = SyntheticDataset(seed=train_dataset_seed)
dataset_for_encoder = mlprogram.utils.data.to_map_style_dataset(
    dataset=SyntheticDataset(seed=0),
    n=100000,
)
test_dataset = mlprogram.utils.data.transform(
    dataset=mlprogram.utils.data.to_map_style_dataset(
        dataset=SyntheticDataset(seed=test_dataset_seed),
        n=n_test_dataset,
    ),
    transform=normalize_dataset
)
valid_dataset = None  # TODO

metrics = {
    "accuracy": mlprogram.metrics.Accuracy(),
    "bleu": mlprogram.languages.python.metrics.Bleu(),
}
