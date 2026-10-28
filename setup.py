from setuptools import setup, find_packages

setup(
    name="mvmgb",
    version="1.0.0",
    author="Rabah Lebsir and Abdelbasset Boukelia",
    author_email="a.boukelia@crbt.dz, rabah.lebsir@univ-guelma.dz",
    url="https://github.com/BoukeliaAbdelbasset/MV-MGB",
    description="Multi-View Meta-Gradient Boosting for Non-Coding RNA Classification",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "biopython>=1.80",
        "joblib>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]
)
