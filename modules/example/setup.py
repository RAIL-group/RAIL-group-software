from setuptools import setup, find_packages


setup(name='example',
      version='1.0.0',
      description='A simple example module',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
