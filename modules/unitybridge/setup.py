from setuptools import setup, find_packages


setup(name='unitybridge',
      version='1.0.0',
      description='Package for TCP communication with Unity3D',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
