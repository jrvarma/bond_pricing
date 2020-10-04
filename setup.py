#!/usr/bin/env python
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='bond_pricing',
      version='0.5.3',
      maintainer='Jayanth R. Varma',
      maintainer_email='jrvarma@gmail.com',
      description='Bond Price with YTM/zero-curve & NPV, IRR, annuities',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/jrvarma/bond_pricing",
      packages=['bond_pricing'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
      ],
      extras_require={
      },
      entry_points={
      },
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: End Users/Desktop",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Programming Language :: Python :: 3",
      ],
      python_requires='>=3.0',
)  # noqa E124
