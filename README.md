
<p align="center">
  <img src="./resources/images/RAIL-group-logo-flat.jpg" width='600px' />
</p>

Open source software provided by the [Robotic Anticipatory Intelligence & Learning (RAIL) Group](https://cs.gmu.edu/~gjstein/) at George Mason University (PI: Prof. Gregory J. Stein), emphasizing capabilities for long-horizon robot planning under uncertainty. In addition to our algorithmic and theoretical contributions, we provide tools for headless visual simulation and procedural environment generation and additionally tutorials in the form of [Jupyter notebooks](./resources/notebooks) that run inside Docker, providing easy and reproducible access to our tools.

This repository can reproduce many of the capabilities and results from the following publications from our lab:
1. Gregory J. Stein, Christopher Bradley, and Nicholas Roy. "Learning over Subgoals for Efficient Navigation of Structured, Unknown Environments." In: Conference on Robot Learning (CoRL). 2018. [paper](http://proceedings.mlr.press/v87/stein18a.html), [talk (14 min)](https://youtu.be/4eHdGUoLlpg). Code Module: [`lsp`](./modules/lsp).
2. Gregory J. Stein. "Generating High-Quality Explanations for Navigation in Partially-Revealed Environments." In: Neural Information Processing Systems (NeurIPS). 2021. [paper](https://proceedings.neurips.cc/paper/2021/hash/926ec030f29f83ce5318754fdb631a33-Abstract.html), [video (13 min)](https://youtu.be/rWxHJJMEPFI), [blog post](https://cs.gmu.edu/~gjstein/2021/11/explainable-navigation-under-uncertainty/). Code Module: [`lsp-xai`](./modules/lsp_xai).
3. Gregory J. Stein, Christopher Bradley, Victoria Preston, and Nicholas Roy. "Enabling Topological Planning with Monocular Vision." In: International Conference on Robotics and Automation (ICRA). 2020. [paper](https://arxiv.org/abs/2003.14368), [talk (10 min)](https://youtu.be/UVZ3UcK6MhI). Code Module: [`vertexnav`](./modules/vertexnav).
4. Raihan Islam Arnob and Gregory J. Stein. "Improving Reliable Navigation under Uncertainty via Predictions Informed by Non-Local Information." International Conference on Intelligent Robots and Systems (IROS). 2023. *paper forthcoming*. Code Module: [`lsp_gnn`](./modules/lsp_gnn).
5. Abhish Khanal and Gregory J. Stein. "Learning Augmented, Multi-Robot Long-Horizon Navigation in Partially Mapped Environments". In: International Conference of Robotics and Automation (ICRA). 2023. [paper](https://arxiv.org/abs/2303.16654). Code Module: [`mrlsp`](./modules/mrlsp).
6. Abhishek Paudel and Gregory J. Stein. "Data-Efficient Policy Selection for Navigation in Partial Maps via Subgoal-Based Abstraction." International Conference on Intelligent Robots and Systems (IROS). 2023. [paper](https://arxiv.org/abs/2304.01094). Code Module: [`lsp_select`](./modules/lsp_select).

See the respective modules for details on each.

## Getting Started Guide

Prerequisities:
- Ubuntu 20.04 or greater.
- Docker with the NVIDIA container runtime. Install Docker via [their website](https://docs.docker.com/get-docker/) and the NVIDIA Docker Runtime via [their official GitHub repository](https://github.com/NVIDIA/nvidia-docker#quickstart).
- GNU Make. Provided as part of the `build-essential` package through apt: `sudo apt-get install build-essential`

Build Steps:
- Clone the repository and `cd` into it.
- Build the Docker container via `make build`.
- [once per system restart] Run `make xhost-activate` to provide the container low-level access to the GPU.

The following top-level commands are then available:
```bash
# Spin up Jupyter, interactive python notebooks, to run our 
# onboarding tutorials and interactive examples. Will build
# the container if necessary. A good starting point.
make notebook

# Run test code for all modules; will build if necessary.
make test
```

Each module---see the list below---provides make targets of their own, often geared towards reproducing results from their respective research publications and contributions. See below and in the module-specific README files for 

### Running Jupyter Notebooks: Tutorials and Demos

We also provide a handful of [Jupyter notebooks](./resources/notebooks) for interactively running code within the Docker container. The `make notebook` command will spin up a Jupyter environment and provide access to both interactive demos and also a few onboarding tutorials used within our lab. Notebooks allow running algorithmic code and procedural environment generation as well as access to the RAIL Sim visual simulator. On GitHub, the notebooks can also be [viewed in the browser](./resources/notebooks) without downloading or building the code.

`make notebook` will spin up a Docker container running Jupyter, which can be accessed from the browser at `http://localhost:8888`.


### Running Tests

Each module comes with tests to verify that the provided code is running correctly. The top-level Make target `test` runs all the tests for each module in the repository within the Docker container. Running `make test` will run all tests via `pytest`. To run only a subset of the tests use the `PYTEST_FILTER` argument: e.g., `make test PYTEST_FILTER=lsp` will run only tests containing the string `lsp`.
