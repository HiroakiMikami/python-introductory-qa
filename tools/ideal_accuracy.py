import argparse

import os
import torch
import tempfile
from tqdm import tqdm

from mlprogram import logging
from mlprogram.entrypoint.configs import load_config, parse_config
from app import types

logger = logging.Logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", required=True, type=str)
    args = parser.parse_args()
    configs = load_config(args.config_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        configs["base_output_dir"] = tmpdir
        configs["output_dir"] = os.path.join(tmpdir, "output")
        configs["device"]["type_str"] = "cpu"
        configs["train_dataset"] = configs["valid_dataset"]
        configs = parse_config(configs, custom_types=types)
        loader = torch.utils.data.DataLoader(
            configs["/train_dataset"],
            batch_size=1,
            collate_fn=configs["/collate_fn"])
        for x in tqdm(loader):
            if x is None:
                print(x)


if __name__ == "__main__":
    main()
