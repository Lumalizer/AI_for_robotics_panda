from setuptools import setup, find_packages

setup(
    name='air_net',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'numpy',
        'pandas',
        'tensorflow',
        'tensorflow_datasets',
        'tensorflow_hub',
        'opencv-python',
        'apache-beam',
        'mlcroissant'
    ],
)
