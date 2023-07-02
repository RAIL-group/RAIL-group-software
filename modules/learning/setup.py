from setuptools import setup, find_packages


setup(name='learning',
      version='1.0.0',
      description='Some shared resources for machine learning.',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'torch'])
