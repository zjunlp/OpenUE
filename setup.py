import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="openue",
    version="0.2.0",
    author="zxlzr",
    author_email="jack16900@gmail.com",
    description="An open toolkit of universal extraction from text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zjunlp/openue",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires='>=3.6'
)
