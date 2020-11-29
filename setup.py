import os
from setuptools import find_packages, setup

requires = [
    "mlprogram @ git+https://github.com/HiroakiMikami/mlprogram.git@d6ff46928b64edd6df2d2c031b5ef245d9d0101f",
    "transformers==3.3.1",
    "scikit-learn==0.23.2",
]

extras = {
    "test": [
        "flake8",
        "autopep8",
        "black",
        "isort",
        "mypy==0.770",
        "timeout-decorator",
        "pytest",
        "pytest-parallel",
    ]
}

setup(
    name="python-introductory-qa",
    version="0.1.0",
    install_requires=requires,
    test_requires=extras["test"],
    extras_require=extras,
    packages=find_packages() + [
        "character_bert",
        os.path.join("character_bert", "modeling"),
        os.path.join("character_bert", "metrics"),
        os.path.join("character_bert", "utils"),
    ],
)
