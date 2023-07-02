from setuptools import setup, find_packages


setup(name='environments',
      version='1.0.0',
      description='Devoted to instantiating simulated environments',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'unitybridge'])
