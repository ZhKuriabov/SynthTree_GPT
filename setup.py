# setup.py
from setuptools import setup, find_packages

setup(
    name="synth_tree",
    version="0.1.5",
    description="Interpretable tree-based model SynthTree for local surrogate modeling",
    author="Evgenii Kuriabov",
    author_email="evgenii@example.com",
    url="https://github.com/ZhKuriabov/SynthTree_GPT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        # добавь другие зависимости, если нужно
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
