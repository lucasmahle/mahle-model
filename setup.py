from setuptools import setup, find_packages

base_packages = [
    "numpy>=1.20.0",
]

spacy_packages = [
    "spacy>=3.2.1",
    'en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.2.0/en_core_web_md-3.2.0.tar.gz'
]

gensim_packages = [
    "gensim>=4.0.0"
]

sklearn_packages = [
    "scikit-learn>=1.0.1"
]

extra_packages = spacy_packages + gensim_packages + sklearn_packages

dev_packages = extra_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mahlemodel",
    packages=find_packages(exclude=[]),
    version="1.0.0",
    author="Lucas S. Mahle",
    author_email="lucassmahle@gmail.com",
    description="MahleModel performs topic Modeling with TF-IDF and K-means.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucasmahle/mahle-model",
    project_urls={
        "Documentation": "https://github.com/lucasmahle/mahle-model/README.md",
        "Source Code": "https://github.com/lucasmahle/mahle-model/",
        "Issue Tracker": "https://github.com/lucasmahle/mahle-model/issues",
    },
    keywords="nlp mahlemodel topic modeling",
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "spacy": spacy_packages,
        "sklearn": sklearn_packages,
        "gensim": gensim_packages
    },
    python_requires='>=3.7',
)