"""Setup script for the research pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gamibench",
    version="0.1.0",
    author="Steven Ngo",
    author_email="svngo@ucsd.edu",
    description="End-to-end evaluation pipeline for MLLMs on 2D-to-3D Origami spatial mapping and reasoning tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stvngo/GamiBench",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gamibench=run:main",
            "gamibench-suite=scripts.run_gamibench_suite:main",
        ],
    },
)
