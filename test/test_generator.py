from app.generator import Generator
import numpy as np
import ast


def test_generation():
    g = Generator(np.random.RandomState(0))
    for _ in range(10000):
        example = g.create()
        ast.parse(example.code)
