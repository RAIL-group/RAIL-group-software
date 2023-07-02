from setuptools import setup, find_packages


setup(name='lsp_gnn',
      version='1.0.0',
      description='Core code for Learned Subgoal Planning using GNN.',
      license="MIT",
      author='Raihan Islam Arnob',
      author_email='rarnob@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
