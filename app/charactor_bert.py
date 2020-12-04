import os
import torch
from mlprogram.languages import Token
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask

import sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, "character_bert"))

# character_bert requires to add project_root to sys.path
from character_bert.download import download_model  # noqa
from character_bert.utils.character_cnn import CharacterIndexer  # noqa
from character_bert.modeling.character_bert import CharacterBertModel  # noqa


class EncodeQuery:
    def __init__(self, max_query_length: int):
        self.max_query_length = max_query_length
        self.indexer = CharacterIndexer()

    def __call__(self, env):
        # Encode for CharacterBERT
        env.states["reference"] = \
            [Token(None, "[CLS]", "[CLS]")] + env.states["reference"] + \
            [Token(None, "[SEP]", "[SEP]")]
        tokens = [token.raw_value for token in env.states["reference"]]
        segmend_ids = [0] * len(tokens)
        input_ids = self.indexer.as_padded_tensor([tokens],
                                                  maxlen=self.max_query_length)[0]
        input_mask = [1] * len(tokens)

        # Zero pad
        padding_length = self.max_query_length - len(input_mask)
        input_mask += [0] * padding_length
        segmend_ids += [0] * padding_length

        env.states["input_ids"] = input_ids
        env.states["input_mask"] = torch.tensor(input_mask)
        env.states["segment_ids"] = torch.tensor(segmend_ids)
        return env


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        download_model(name="bert-base-uncased")
        self.model = CharacterBertModel.from_pretrained(
            os.path.join("pretrained-models", "bert-base-uncased"))

    def forward(self, env):
        seqence_ouptut, pooled_output = self.model(
            input_ids=env.states["input_ids"],
            attention_mask=env.states["input_mask"],
            token_type_ids=env.states["segment_ids"])
        padded_sequence_output = PaddedSequenceWithMask(
            seqence_ouptut.permute(1, 0, 2),
            env.states["input_mask"].permute(1, 0))
        # TODO should we use pooled_output?
        env.states["nl_query_features"] = padded_sequence_output
        env.states["reference_features"] = padded_sequence_output
        return env


def Model(model_dir: str):
    return CharacterBertModel.from_pretrained(model_dir)
