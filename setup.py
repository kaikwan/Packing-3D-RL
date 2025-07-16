from setuptools import setup, find_packages

setup(
    name="packing3d",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="3D bin packing solver using heuristic search and RL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yang Ding",
    author_email="yangding19thu@163.com",
    url="https://github.com/yang-d19/Packing-3D-RL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "packing3d=packing3d.test:main",
        ],
    },
)