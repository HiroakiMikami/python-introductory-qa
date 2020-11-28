import numpy as np
from mlprogram import Environment
from mlprogram.languages import Token
from app.charactor_bert import EncodeQuery, Extractor


def test_encode():
    encoder = EncodeQuery(128)
    out = encoder(Environment(
        states={"reference": [Token(None, "print", "print"),
                              Token(None, "___var@1___", "`x`")]}
    ))
    assert out.states["input_ids"].shape == (128, 50)
    assert out.states["segment_ids"].shape == (128,)
    assert out.states["input_mask"].shape == (128,)
    assert np.all(out.states["segment_ids"].numpy() == 0)
    assert np.all(out.states["input_mask"][:4].numpy() == 1)
    assert np.all(out.states["input_mask"][4:].numpy() == 0)


def test_extractor():
    encoder = EncodeQuery(128)
    extractor = Extractor()
    out = encoder(Environment(
        states={"reference": [Token(None, "print", "print"),
                              Token(None, "___var@1___", "`x`")]}
    ))
    out.states["input_ids"] = out.states["input_ids"].unsqueeze(0)
    out.states["segment_ids"] = out.states["segment_ids"].unsqueeze(0)
    out.states["input_mask"] = out.states["input_mask"].unsqueeze(0)
    out = extractor(out)
    assert out.states["reference_features"].data.shape == (128, 1, 768)
    assert not out.states["reference_features"].data.requires_grad
    assert np.all(out.states["reference_features"].mask[:4].numpy() == 1)
    assert np.all(out.states["reference_features"].mask[4:].numpy() == 0)
