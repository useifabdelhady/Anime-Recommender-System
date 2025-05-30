from setuptools import setup, find_packages

with open("requirements.txt") as f:
	requirements = f.read().splitlines()

setup(
	name="Hotel_Reservation",
	version="0.1",
	author="Yousef",
    packages=find_packages(),
    install_requires=requirements,
)