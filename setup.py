import setuptools

setuptools.setup(
    name="maga",
    version="0.1",
    author="Group 7",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "wandb",
        "scipy",
        "scikit-learn",
        "pandas",
    ],
)