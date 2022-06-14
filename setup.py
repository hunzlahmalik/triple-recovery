from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='triplerecovery',
    version='0.0.1',
    description='Triple Recovery paper implementation in python',
    py_modules=['triplerecovery'],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/crackaf/triple-recovery",
    author="Hunzlah Malik",
    author_email="hunzlahmalik@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",

    install_requires=[
        "numpy",
        "scipy",
    ],
    extra_requires={
        "test": [
            "pytest",
        ]
    }
)
