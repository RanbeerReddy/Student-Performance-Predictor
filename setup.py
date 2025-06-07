from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            if line and line != HYPHEN_E_DOT:
                requirements.append(line)
    return requirements 
      


setup(
    name='mlproject',
    version='0.0.1', 
    author='Ranbeer',
    author_email='reddyranbeer@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)