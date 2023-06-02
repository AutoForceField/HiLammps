from setuptools import find_packages, setup

with open("hilammps/version.py") as f:
    _version: dict[str, str] = {}
    exec(f.read(), _version)
    __version__ = _version["__version__"]


setup(
    name="hilammps",
    version=__version__,
    author="Amir Hajibabaei",
    author_email="autoforcefield@gmail.com",
    description="Higher level interface for LAMMPS",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy"],
    url="https://github.com/AutoForceField/HILammps",
    license="MIT",
)