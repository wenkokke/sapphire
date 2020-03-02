import setuptools

setuptools.setup(
    name="sapphire",
    version="0.0.1",
    author="Wen Kokke",
    author_email="wen.kokke@ed.ac.uk",
    description="A library for translating TensorFlow models to Z3",
    license="Peer Production License",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wenkokke/sapphire",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha"
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    data_files = [("", ["LICENSE.txt"])]
)
