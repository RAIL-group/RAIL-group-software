from setuptools import setup, find_packages


setup(name='mrlsp',
      version='1.0.0',
      description='A simple mrlsp using lsp module',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
