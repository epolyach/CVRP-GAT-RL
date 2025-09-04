from setuptools import setup, find_packages

setup(
    name="gat_rl_vrp",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.0.0",
        "pandas",
        "numpy",
        "scipy",
        "tensorboard",
    ],
)
