from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name='elite',
    version='0.0.1',
    packages=find_packages(),
    install_requires=required_packages,
)
