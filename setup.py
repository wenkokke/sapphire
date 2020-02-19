import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sapphire",
    version="0.0.1",
    author="Wen Kokke",
    author_email="wen.kokke@ed.ac.uk",
    description="A library for translating TensorFlow models to Z3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wenkokke/sapphire",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha"
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
