from setuptools import find_packages
from distutils.core import setup

setup(
    name='pycgp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'joblib',
        'numpy',
        'pytest'
    ],
    license='MIT',
)
