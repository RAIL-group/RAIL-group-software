make build

echo $DATA_BASE_DIR
echo $UNITY_DIR

# Maze Experiments
make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_0SG \
     XAI_LEARNING_SEED=8616 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=0 -j4

make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_4SG \
     XAI_LEARNING_SEED=8616 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=4 -j4
make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_allSG \
     XAI_LEARNING_SEED=8616 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=-1 -j4

make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_rs1_0SG \
     XAI_LEARNING_SEED=8617 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=0 -j4
make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_rs1_4SG \
     XAI_LEARNING_SEED=8617 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=4 -j4
make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_rs1_allSG \
     XAI_LEARNING_SEED=8617 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=-1 -j4


make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_rs2_0SG \
     XAI_LEARNING_SEED=8618 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=0 -j4
make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_rs2_4SG \
     XAI_LEARNING_SEED=8618 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=4 -j4
make xai-maze \
     MAZE_XAI_BASENAME=lsp/maze_tmp \
     EXPERIMENT_NAME=postsub_1k_002xent_rs2_allSG \
     XAI_LEARNING_SEED=8618 \
     XAI_LEARNING_XENT_FACTOR=0.02 \
     SP_LIMIT_NUM=-1 -j4


make xai-floorplan \
     FLOORPLAN_XAI_BASENAME=lsp/floorplan_interp_oriented \
     EXPERIMENT_NAME=postsub_1k_1xent_rs1_allSG \
     XAI_LEARNING_SEED=8617 \
     XAI_LEARNING_XENT_FACTOR=1.0 \
     SP_LIMIT_NUM=-1 -j4
make xai-floorplan \
     FLOORPLAN_XAI_BASENAME=lsp/floorplan_interp_oriented \
     EXPERIMENT_NAME=postsub_1k_1xent_rs1_4SG \
     XAI_LEARNING_SEED=8617 \
     XAI_LEARNING_XENT_FACTOR=1.0 \
     SP_LIMIT_NUM=4 -j4
make xai-floorplan \
     FLOORPLAN_XAI_BASENAME=lsp/floorplan_interp_oriented \
     EXPERIMENT_NAME=postsub_1k_1xent_rs1_0SG \
     XAI_LEARNING_SEED=8617 \
     XAI_LEARNING_XENT_FACTOR=1.0 \
     SP_LIMIT_NUM=0 -j4
