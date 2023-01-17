from setuptools import setup, find_packages

setup(
    name="homevision-models",
    version="0.1",
    description="HomeVision - Take Home",
    author="Jonathan Loscalzo",
    packages=find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
)
