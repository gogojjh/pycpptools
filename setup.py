from setuptools import setup, find_packages

setup(
    name="pycpptools",
    version="0.1",
    packages=find_packages(),
    install_requires=open("requirements.txt", "r").read().split("\n"),
)