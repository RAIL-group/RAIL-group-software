from setuptools import setup, find_packages

setup(name='lsp_xai',
      version='1.0.0',
      description='Core code for Learned Subgoal Planning.',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
