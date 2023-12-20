#!/usr/bin/env python
from setuptools import setup
from os import environ

dependencies = ['numpy', 'pandas']
if 'no_scipy' not in environ:
    dependencies += ['scipy']

setup(install_requires=dependencies)
