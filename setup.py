from setuptools import setup, find_packages

setup(
    name='ai_for_robotics_panda',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'matplotlib',
        'numpy',
    ],
)