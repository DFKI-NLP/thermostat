from setuptools import find_packages, setup
# new change 
from sys import platform

if platform == "win32":
    jsonnet = "jsonnet-binary"
else:
    jsonnet = "jsonnet"
# new change

REQUIRED_PKGS = [
    "captum>=0.3",
    "datasets>=1.5",
    jsonnet,
    "numpy>=1.22",
    "overrides",
    "pandas",
    "protobuf",
    "pytorch-ignite",
    "scipy",
    "sentencepiece",
    "scikit-learn",
    "spacy>=3.0",
    "torch",
    "tqdm>=4.49",
    "transformers>=4.5",
]

setup(
    name="thermostat-datasets",
    version="1.1.0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Collection of NLP model explanations and accompanying analysis tools",
    long_description="Thermostat is a large collection of NLP model explanations and accompanying analysis tools. "
                     "Combines explainability methods from the captum library with Hugging Face's datasets and "
                     "transformers. Mitigates repetitive execution of common experiments in Explainable NLP and thus "
                     "reduces the environmental impact and financial roadblocks. Increases comparability and "
                     "replicability of research. Reduces the implementational burden.",
    author="DFKI-NLP",
    author_email="nils.feldhus@dfki.de",
    url="https://github.com/DFKI-NLP/thermostat",
    download_url="https://github.com/DFKI-NLP/thermostat/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=REQUIRED_PKGS,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="explainability heatmaps feature-attribution natural-language-processing",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
