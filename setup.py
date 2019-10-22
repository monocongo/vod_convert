import pathlib
from setuptools import setup

# the directory containing this file
BASE_DIR = pathlib.Path(__file__).parent

# the text of the README file
README = (BASE_DIR / "README.md").read_text()

setup(
    name="vod_convert",
    version="0.0.1",
    url="https://github.com/monocongo/vod_convert",
    license="MIT",
    author="Adams, James",
    author_email="monocongo@gmail.com",
    description=(
        "Toolkit to facilitate conversion between popular visual object "
        "detection annotation formats."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    packages=["vod_convert"],
    include_package_data=True,
    install_requires=[
    ],
    keywords=[
        "object detection",
        "image classification",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={
        "console_scripts": [
            "vod_convert = vod_convert.__main__:main",
        ],
    }
)
