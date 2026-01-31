"""
Hyper-Mnemosyne: Advanced Neural Memory Architecture
Minimal setup for development installation
"""

from setuptools import setup, find_packages

setup(
    name="hyper-mnemosyne",
    version="0.1.0",
    description="Hyper-Mnemosyne: Mamba-2 + Titans Memory + JEPA",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "optional": [
            "bitsandbytes>=0.41.0",  # For 8-bit AdamW
            "mamba-ssm>=1.0.0",      # For Mamba-2 kernels
        ]
    },
)
