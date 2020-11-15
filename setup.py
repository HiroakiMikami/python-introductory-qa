from setuptools import find_packages, setup

requires = [
    "mlprogram @ git+https://github.com/HiroakiMikami/mlprogram.git@bbd446faaba5aac4cfc654a03df00b7ae28241ea"
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
    packages=find_packages(),
)
