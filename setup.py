from setuptools import setup, find_packages
from typing import List

EDITABLE_INSTALL = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Read requirements.txt and return a list of package names."""
    requirements = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and line != EDITABLE_INSTALL:
                requirements.append(line)
    return requirements


setup(
    name="fraudproject",
    version="0.1.0",
    author="dip",
    description="Credit Card Fraud Detection — End-to-End ML Project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
