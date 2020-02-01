from setuptools import setup, find_packages
import setuptools

setup(
    name='lineage_detection_robot',
    version='0.1.0',
    description='Families In the WIld: A Kinship Recogntion Toolbox.',
    long_description="none",
    author='Jack Horsburgh, Ciaran Johnson, Ishan Parikh',
    author_email='s1627278@sms.ed.ac.uk',
    url='https://github.com/LineageDetectingRobots/MLP-Project',
    packages=setuptools.find_packages(),
    license="none",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)