# MR-LSP: Multi-Robot Learning over Subgoals Planning

Coordinated multi-robot goal directed navigation in partially mapped environments. This module contains the core code for learning-augmented long-horizon naivgation for a multi-robot team, itself an extension of the Learning over Subgoals Planning paradigm for multiple robots. The MR-LSP approach is found in the following paper, which contains algorithmic details:

Abhish Khanal and Gregory J. Stein. "Learning Augmented, Multi-Robot Long-Horizon Navigation in Partially Mapped Environments". In: International Conference of Robotics and Automation (ICRA). 2023. [paper](https://arxiv.org/abs/2303.16654).

```bibtex
@inproceedings{khanal2023learning,
      title={Learning Augmented, Multi-Robot Long-Horizon Navigation in Partially Mapped Environments}, 
      author={Abhish Khanal and Gregory J. Stein},
      booktitle={International Conference on Robotics and Automation (ICRA)}
      year={2023},
}
```

## Usage

Note: `make build` (see top-level README) must be successfully run before running the following commands.
The `Makefile.mk` provides multiple targets for reproducing results and visualize planners.
## To visualize planners:
- <code> make visualize-office PLANNER=*planner_name* SEED=*seed* NUM_ROBOTS=*num_robots*</code> visualizes the planner for different number of robots in office environment. Here, *planner_name* can be {optimistic, baseline, mrlsp}, and seed controls the random seed. [Note: Changing the seed also changes the map.]
- <code> make visualize-maze PLANNER=*planner_name* SEED=*seed* NUM_ROBOTS=*num_robots*</code> does the same for maze environment.

## To run experiments from the paper:

- `make mrlsp-maze` will run the experiments for all the planner in the maze environment. Note that, if trained model is not present, it first trains the neural network and then runs the experiments in maze environment. 
- `make mrlsp-office` will run experiments for all the planner in `office2` environment. 
