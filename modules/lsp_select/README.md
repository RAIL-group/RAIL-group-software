# Policy Selection for Learning over Subgoals Planning

This module provides the implementation of all experiments in the paper:

Abhishek Paudel and Gregory J. Stein. "Data-Efficient Policy Selection for Navigation in Partial Maps via Subgoal-Based Abstraction." International Conference on Intelligent Robots and Systems (IROS). 2023. [paper](https://arxiv.org/abs/2304.01094)

```bibtex
@inproceedings{paudel2023lspselect,
  title={Data-Efficient Policy Selection for Navigation in Partial Maps via Subgoal-Based Abstraction},
  author={Paudel, Abhishek and Stein, Gregory J.},
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

## Usage
### Reproducing Results
Note: `make build` (see top-level README) must be successfully run before running the following commands.

The `Makefile.mk` provides multiple targets for reproducing the results.

- `make lsp-policy-selection-maze` generates results for simulated maze environments. It first generates all the training data in different variations of maze environments (see paper for details), trains a neural network in each of these environments, and deploys LSP policies using the trained neural networks (as well as a non-learned policy) in all of the maze environments. Finally, it runs our policy selection algorithm to reproduce the results shown in the paper.

- `make lsp-policy-selection-office` generates the results corresponding to office experiments described in the paper. The process is similar to above.

- `make lsp-policy-selection-check`	runs very minimal policy selection experiments for maze environments (with only few maps) to verify that everything works as intended.

All targets run in single-threaded mode by default, however data generation and offline-replay costs generation can be run on multiple seeds in parallel. For example `make lsp-select-gen-data-mazeA -j3` will run three concurrent instances of data generation in `mazeA` and `make lsp-select-offline-replay-costs-maze -j3` will run three concurrent deployment experiments in maze environments.
