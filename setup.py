from setuptools import find_packages, setup

setup(name="voc_utils",
      version="0.0",
      description="A python utility for loading data in Pascal VOC format",
      author="Michele Pratusevich",
      author_email='mprat@alum.mit.edu',
      platforms=["osx"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="BSD",
      url="http://github.com/mprat/pascal-voc-python",
      packages=find_packages(),
      install_requires=[i.strip() for i in open("requirements.txt").readlines()]
      )
