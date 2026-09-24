from setuptools import setup, find_packages

setup(
    name="medical-tumor-pipeline",
    version="1.0.0",
    description="BraTS 2023 Multimodal Brain Tumor Detection and Classification Pipeline",
    author="suzirz",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "monai>=1.3.0",
        "timm>=0.9.12",
        "nibabel>=5.2.0",
        "SimpleITK>=2.3.1",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "streamlit>=1.28.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
