# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="bda case challenge",
    version="0.1.0",
    description="dummy description",
    author="dummy author",
    author_email="dummy@mail",
    packages=find_packages(exclude=("tests", "docs")),
)