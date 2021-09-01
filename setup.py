
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openue",
    version="0.0.4",
    author="zxlzr",
    author_email="jack16900@gmail.com",
    description="An open toolkit of universal extraction from text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zjunlp/openue",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
