from setuptools import setup, find_packages

setup(
    name='knn_attention',
    version='0.1.0',
    description='Code for the kNN attention paper.',
    url='https://github.com/sansui-123/knn_attention',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'faiss',
    ],
)
