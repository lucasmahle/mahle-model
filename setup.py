from setuptools import setup, find_packages

base_packages = [
    "numpy>=1.20.0",
    "gensim>=4.0.0",
    "scikit-learn>=1.0.1",
    "spacy>=3.6,<3.7",
    'en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.6.0/en_core_web_md-3.6.0.tar.gz'
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mahlemodel",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    version="2.1.0",
    author="Lucas Mahle",
    author_email="lucassmahle@gmail.com",
    description="MahleModel performs topic modeling with TF-IDF and K-means.",
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
    python_requires='>=3.7',
)