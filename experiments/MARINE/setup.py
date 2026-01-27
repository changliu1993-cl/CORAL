from setuptools import setup, find_packages

setup(
    name="marine",
    version="0.1.0",
    description="MARINE: Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "numpy>=1.24",
        "Pillow>=9.0",
        "tqdm",
        "nltk>=3.6",
        "scikit-learn>=1.0",
        "matplotlib",
        "jsonlines",
        "pyyaml",
        "opencv-python"
    ],
    python_requires=">=3.8",
)
