"""Setup script for goai-bench package."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="goai-bench",
    version="0.1.0",
    description=(
        "GO AI Bench — Open benchmarking platform for low-resource "
        "Burkinabè and West African NLP languages"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="GO AI Corporation",
    license="Apache-2.0",
    url="https://github.com/GO-AI-CORPORATION/goai-bench",
    project_urls={
        "HuggingFace Space": "https://huggingface.co/spaces/goaicorp/goai-bench",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "huggingface-hub>=0.22.0",
        "sacrebleu>=2.4.0",
        "jiwer>=3.0.3",
        "matplotlib>=3.8.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "filelock>=3.13.0",
        "click>=8.1.7",
    ],
    extras_require={
        "full": [
            "unbabel-comet>=2.2.0",
            "librosa>=0.10.1",
            "soundfile>=0.12.1",
            "openai-whisper>=20231117",
            "accelerate>=0.29.0",
            "sentencepiece>=0.1.99",
            "pymcd>=0.1.0",
        ],
        "dev": [
            "pytest>=8.1.0",
            "pytest-cov>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "goai-bench=goai_bench.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
