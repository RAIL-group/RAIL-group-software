
MAZE_XAI_BASENAME ?= maze
FLOORPLAN_XAI_BASENAME ?= floorplan
SP_LIMIT_NUM ?= -1
LSP-XAI_LEARNING_SEED ?= 8616
LSP-XAI_LEARNING_XENT_FACTOR ?= 1
LSP-XAI_UNITY_BASENAME ?= $(UNITY_BASENAME)

## ==== Core arguments ====

TEST_ADDITIONAL_ARGS += --lsp-xai-maze-net-0SG-path=/resources/testing/lsp_xai_maze_0SG.ExpNavVisLSP.pt

SIM_ROBOT_ARGS ?= --step_size 1.8 \
		--num_primitives 32 \
		--field_of_view_deg 360
INTERP_ARGS ?= --summary_frequency 100 \
		--num_epochs 1 \
		--learning_rate 2.0e-2 \
		--batch_size 4

## ==== Maze Arguments and Experiments ====
MAZE_CORE_ARGS ?= --unity_path /unity/$(LSP-XAI_UNITY_BASENAME).x86_64 \
		--map_type maze \
		--base_resolution 0.3 \
		--inflation_rad 0.75 \
		--laser_max_range_m 18 \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/
MAZE_DATA_GEN_ARGS = $(MAZE_CORE_ARGS) --logdir /data/lsp_xai/$(MAZE_XAI_BASENAME)/training/data_gen
MAZE_EVAL_ARGS = $(MAZE_CORE_ARGS) --logdir /data/lsp_xai/$(MAZE_XAI_BASENAME)/training/$(EXPERIMENT_NAME) \
		--logfile_name logfile_final.txt

lsp-xai-maze-dir = $(DATA_BASE_DIR)/lsp_xai/$(MAZE_XAI_BASENAME)

# Initialize the Learning
lsp-xai-maze-init-learning = $(lsp-xai-maze-dir)/training/data_gen/ExpNavVisLSP.init.pt
$(lsp-xai-maze-init-learning):
	@echo "Writing the 'initial' neural network: $@"
	@mkdir -p $(lsp-xai-maze-dir)/training/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_DATA_GEN_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_init_learning

# Generate Data
lsp-xai-maze-data-gen-seeds = $(shell for ii in $$(seq 1000 1299); do echo "$(lsp-xai-maze-dir)/data_collect_plots/learned_planner_$$ii.png"; done)
$(lsp-xai-maze-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-xai-maze-data-gen-seeds): $(lsp-xai-maze-init-learning)
	@echo "Generating Data: $@"
	@$(call xhost_activate)
	@$(call arg_check_unity)
	@rm -f $(lsp-xai-maze-dir)/lsp_data_$(seed).*.csv
	@mkdir -p $(lsp-xai-maze-dir)/data
	@mkdir -p $(lsp-xai-maze-dir)/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_DATA_GEN_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
	 	--do_data_gen \
	 	--current_seed $(seed)

# Train the Network
lsp-xai-maze-train-learning = $(lsp-xai-maze-dir)/training/$(EXPERIMENT_NAME)/ExpNavVisLSP.final.pt
$(lsp-xai-maze-train-learning): $(lsp-xai-maze-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--sp_limit_num $(SP_LIMIT_NUM) \
		--learning_seed $(LSP-XAI_LEARNING_SEED) \
		--xent_factor $(LSP-XAI_LEARNING_XENT_FACTOR) \
		--do_train

# Evaluate Performance
lsp-xai-maze-eval-seeds = $(shell for ii in $$(seq 11000 11099); do echo "$(lsp-xai-maze-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$$ii.png"; done)
$(lsp-xai-maze-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-xai-maze-eval-seeds): $(lsp-xai-maze-train-learning)
	@echo "Evaluating Performance: $@"
	@$(call xhost_activate)
	@$(call arg_check_unity)
	@mkdir -p $(lsp-xai-maze-dir)/results/$(EXPERIMENT_NAME)
	$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_eval \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--current_seed $(seed)


## ==== University Building (Floorplan) Environment Experiments ====
FLOORPLAN_CORE_ARGS ?= --unity_path /unity/$(LSP-XAI_UNITY_BASENAME).x86_64 \
		--map_type ploader \
		--base_resolution 0.6 \
		--inflation_radius_m 1.5 \
		--laser_max_range_m 72 \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/
FLOORPLAN_DATA_GEN_ARGS ?= $(FLOORPLAN_CORE_ARGS) \
		--map_file /resources/university_building_floorplans/train/*.pickle \
		--logdir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/training/data_gen
FLOORPLAN_EVAL_ARGS ?= $(FLOORPLAN_CORE_ARGS) \
		--map_file /resources/university_building_floorplans/test/*.pickle \
		--logdir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/training/$(EXPERIMENT_NAME) \
		--logfile_name logfile_final.txt
lsp-xai-floorplan-dir = $(DATA_BASE_DIR)/lsp_xai/$(FLOORPLAN_XAI_BASENAME)


# Initialize the Learning
lsp-xai-floorplan-init-learning = $(lsp-xai-floorplan-dir)/training/data_gen/ExpNavVisLSP.init.pt
$(lsp-xai-floorplan-init-learning):
	@echo "Writing the 'initial' neural network [Floorplan: $(FLOORPLAN_XAI_BASENAME)]"
	@mkdir -p $(lsp-xai-floorplan-dir)/training/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_DATA_GEN_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_init_learning

# Generate Data
lsp-xai-floorplan-data-gen-seeds = $(shell for ii in $$(seq 1000 1009); do echo "$(lsp-xai-floorplan-dir)/data_collect_plots/learned_planner_$$ii.png"; done)
$(lsp-xai-floorplan-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-xai-floorplan-data-gen-seeds): $(lsp-xai-floorplan-init-learning)
	@echo "Generating Data: $@"
	@$(call xhost_activate)
	@rm -f $(lsp-xai-floorplan-dir)/lsp_data_$(seed).*.csv
	@mkdir -p $(lsp-xai-floorplan-dir)/data
	@mkdir -p $(lsp-xai-floorplan-dir)/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_DATA_GEN_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
	 	--do_data_gen \
	 	--current_seed $(seed)

# Train the Network
lsp-xai-floorplan-train-learning = $(lsp-xai-floorplan-dir)/training/$(EXPERIMENT_NAME)/ExpNavVisLSP.final.pt
$(lsp-xai-floorplan-train-learning): $(lsp-xai-floorplan-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--sp_limit_num $(SP_LIMIT_NUM) \
		--learning_seed $(LSP-XAI_LEARNING_SEED) \
		--xent_factor $(LSP-XAI_LEARNING_XENT_FACTOR) \
		--do_train

# Evaluate Performance
lsp-xai-floorplan-eval-seeds = $(shell for ii in $$(seq 11000 11999); do echo "$(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$$ii.png"; done)
$(lsp-xai-floorplan-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-xai-floorplan-eval-seeds): $(lsp-xai-floorplan-train-learning)
	@echo "Evaluating Performance: $@"
	@$(call xhost_activate)
	@$(call arg_check_unity)
	@mkdir -p $(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)
	$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_eval \
		--current_seed $(seed) \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \


# Some helper targets to run code individually
lsp-xai-floorplan-intervene-seeds-4SG = $(shell for ii in 11304 11591 11870 11336 11245 11649 11891 11315 11069 11202 11614 11576 11100 11979 11714 11430 11267 11064 11278 11367 11193 11670 11385 11180 11923 11195 11642 11462 11010 11386 11913 11103 11474 11855 11823 11641 11408 11899 11449 11393 11041 11435 11101 11610 11422 11546 11048 11070 11699 11618; do echo "$(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$${ii}_intervened_4SG.png"; done)
lsp-xai-floorplan-intervene-seeds-allSG = $(shell for ii in 11304 11591 11870 11336 11245 11649 11891 11315 11069 11202 11614 11576 11100 11979 11714 11430 11267 11064 11278 11367 11193 11670 11385 11180 11923 11195 11642 11462 11010 11386 11913 11103 11474 11855 11823 11641 11408 11899 11449 11393 11041 11435 11101 11610 11422 11546 11048 11070 11699 11618; do echo "$(lsp-xai-floorplan-dir)/results/$(EXPERIMENT_NAME)/learned_planner_$${ii}_intervened_allSG.png"; done)
$(lsp-xai-floorplan-intervene-seeds-4SG): $(lsp-xai-floorplan-train-learning)
	@mkdir -p $(DATA_BASE_DIR)/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_intervene \
		--sp_limit_num 4 \
	 	--current_seed $(shell echo $@ | grep -Eo '[0-9]+' | tail -2 | head -1) \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--logfile_name logfile_intervene_4SG.txt

$(lsp-xai-floorplan-intervene-seeds-allSG): $(lsp-xai-floorplan-train-learning)
	@mkdir -p $(DATA_BASE_DIR)/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
		$(FLOORPLAN_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		$(INTERP_ARGS) \
		--do_intervene \
	 	--current_seed $(shell echo $@ | grep -Eo '[0-9]+' | tail -1) \
		--save_dir /data/lsp_xai/$(FLOORPLAN_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--logfile_name logfile_intervene_allSG.txt \

## ==== Results & Plotting ====
.PHONY: lsp-xai-process-results
lsp-xai-process-results:
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)/logfile_final.txt \
		--output_image_file /data/tmp.png
	@echo "==== Maze Results ===="
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(MAZE_XAI_BASENAME)/results/base_allSG/logfile_final.txt \
			/data/$(MAZE_XAI_BASENAME)/results/base_4SG/logfile_final.txt \
			/data/$(MAZE_XAI_BASENAME)/results/base_0SG/logfile_final.txt \
		--output_image_file /data/maze_results.png
	@echo "==== Floorplan Results ===="
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(FLOORPLAN_XAI_BASENAME)/results/base_allSG/logfile_final.txt \
			/data/$(FLOORPLAN_XAI_BASENAME)/results/base_4SG/logfile_final.txt \
			/data/$(FLOORPLAN_XAI_BASENAME)/results/base_0SG/logfile_final.txt \
		--output_image_file /data/floorplan_results.png
	@echo "==== Floorplan Intervention Results ===="
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_results \
		--data_file /data/$(FLOORPLAN_XAI_BASENAME)/results/base_4SG/logfile_intervene_4SG.txt \
			/data/$(FLOORPLAN_XAI_BASENAME)/results/base_4SG/logfile_intervene_allSG.txt \
		--do_intervene \
		--xpassthrough $(XPASSTHROUGH) \
		--output_image_file /data/floorplan_intervene_results.png \

.PHONY: lsp-xai-explanations
lsp-xai-explanations:
	@mkdir -p $(DATA_BASE_DIR)/lsp_xai/explanations/
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
	 	$(MAZE_EVAL_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
		--do_explain \
		--explain_at 3 \
	 	--sp_limit_num -1 \
	  	--current_seed 1037 \
	 	--save_dir /data/lsp_xai/explanations/ \
	 	--logdir /data/lsp_xai/maze/training/dbg099
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
	 	$(FLOORPLAN_EVAL_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
	 	--do_explain \
	 	--explain_at 289 \
	 	--sp_limit_num 4 \
	  	--current_seed 11591 \
	 	--save_dir /data/explanations/ \
	 	--logdir /data/$(FLOORPLAN_XAI_BASENAME)/training/base_4SG

lsp-xai-assip: seed ?= 11035
lsp-xai-assip:
	mkdir -p $(DATA_BASE_DIR)/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)
	mkdir -p $(DATA_BASE_DIR)/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)/data
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.explainability_train_eval \
	 	$(MAZE_EVAL_ARGS) \
	 	$(SIM_ROBOT_ARGS) \
	 	$(INTERP_ARGS) \
	 	--do_intervene \
	  	--current_seed $(seed) \
	 	--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.interpret_datum \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		--datum_name dat_$(seed)_intervention.combined.pgz \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)/learned_planner_$(seed)_intervention_weights.before.pt \
		--image_base_name learned_planner_$(seed)_attr_before
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.interpret_datum \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		--datum_name dat_$(seed)_intervention.combined.pgz \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)/learned_planner_$(seed)_intervention_weights.after.pt \
		--image_base_name learned_planner_$(seed)_attr_after
	@$(DOCKER_PYTHON) -m lsp_xai.scripts.interpret_datum \
		$(MAZE_EVAL_ARGS) \
		$(SIM_ROBOT_ARGS) \
		--datum_name dat_$(seed)_intervention.combined.pgz \
		--save_dir /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/lsp_xai/maze_scratch_new_sim/training/$(EXPERIMENT_NAME)/ExpNavVisLSP.final.pt \
		--image_base_name learned_planner_$(seed)_attr_smart

## ==== Some helper targets to run code individually ====
# Maze
lsp-xai-maze-data-gen: $(lsp-xai-maze-data-gen-seeds)
lsp-xai-maze-train: $(lsp-xai-maze-train-learning)
lsp-xai-maze-eval: $(lsp-xai-maze-eval-seeds)
lsp-xai-maze: lsp-xai-maze-eval
	$(MAKE) lsp-xai-maze-results

lsp-xai-maze-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		--data_file /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/$(EXPERIMENT_NAME)/logfile_final.txt \
		--output_image_file /data/lsp_xai/$(MAZE_XAI_BASENAME)/results/results_$(EXPERIMENT_NAME).png

# Floorplan
lsp-xai-floorplan-data-gen: $(lsp-xai-floorplan-data-gen-seeds)
lsp-xai-floorplan-train: $(lsp-xai-floorplan-train-learning)
lsp-xai-floorplan-eval: $(lsp-xai-floorplan-eval-seeds)
lsp-xai-floorplan: lsp-xai-floorplan-eval

lsp-xai-floorplan-data-gen: $(lsp-xai-floorplan-data-gen-seeds)
lsp-xai-floorplan-intervene: $(lsp-xai-floorplan-intervene-seeds-allSG) $(lsp-xai-floorplan-intervene-seeds-4SG)
