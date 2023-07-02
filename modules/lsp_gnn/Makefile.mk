help::
	@echo "Learned Subgoal Planning using GNN(lsp-gnn):"
	@echo "  lsp-gnn-jshaped		Runs the 'J-intersection' experiments."
	@echo "  lsp-gnn-parallel		Runs the 'Parallel hallway' experiments."
	@echo "  lsp-gnn-floorplans		Runs the 'University building floorplans' experiments."
	@echo "  lsp-gnn				Runs the experiments in all 3 environments."

LSP_WEIGHT ?= 1
LOC_WEIGHT ?= 1
LOSS ?= l1
RPW ?= 1
INPUT_TYPE ?= wall_class
LSP_GNN_CORE_ARGS ?= --unity_path /unity/$(UNITY_BASENAME).x86_64 \
		--save_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/ \
		--base_resolution 0.4 \
		--inflation_radius_m 0.75 \
		--field_of_view_deg 360 \
		--laser_max_range_m 12 \
		--pickle_directory data_pickles

LSP_GNN_TRAINING_ARGS ?= $(LSP_GNN_CORE_ARGS) \
		--lsp_weight $(LSP_WEIGHT) \
		--loc_weight $(LOC_WEIGHT) \
		--loss $(LOSS) \
		--relative_positive_weight $(RPW) \
		--input_type $(INPUT_TYPE)

#---=== J-Intersection Environment Experiments ===---#
LSP_GNN_JSHAPED_BASENAME ?= lsp_gnn/jshaped
LSP_GNN_JSHAPED_NUM_TRAINING_SEEDS ?= 200
LSP_GNN_JSHAPED_NUM_TESTING_SEEDS ?= 50
LSP_GNN_JSHAPED_NUM_EVAL_SEEDS ?= 100

lsp-gnn-jshaped-data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(LSP_GNN_JSHAPED_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/data_completion_logs/data_training_$${ii}.txt"; done) \
	$(shell for ii in $$(seq 30000 $$((30000 + $(LSP_GNN_JSHAPED_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/data_completion_logs/data_testing_$${ii}.txt"; done)

$(lsp-gnn-jshaped-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-gnn-jshaped-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-gnn-jshaped-data-gen-seeds):
	@echo "Generating Data [$(LSP_GNN_JSHAPED_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/data_completion_logs
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/error_logs
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.gen_data \
		$(LSP_GNN_CORE_ARGS) \
		--map_type jshaped \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/

lsp-gnn-jshaped-autoencoder-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt
$(lsp-gnn-jshaped-autoencoder-train-file): $(lsp-gnn-jshaped-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 28000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 7000 \
		--save_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/ 

lsp-gnn-jshaped-cnn-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt
$(lsp-gnn-jshaped-cnn-train-file): $(lsp-gnn-jshaped-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 16000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 4000 \
		--autoencoder_network_file /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_cnn_lsp \
		--save_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/ 

lsp-gnn-jshaped-marginal-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt 
$(lsp-gnn-jshaped-marginal-train-file): $(lsp-gnn-jshaped-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 16000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 4000 \
		--autoencoder_network_file /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_marginal_lsp \
		--save_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/ 

lsp-gnn-jshaped-eval-seeds = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_GNN_JSHAPED_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/results/$(EXPERIMENT_NAME)/jshaped_learned_$${ii}.png"; done)
$(lsp-gnn-jshaped-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-gnn-jshaped-eval-seeds): $(lsp-gnn-jshaped-cnn-train-file)
$(lsp-gnn-jshaped-eval-seeds): $(lsp-gnn-jshaped-marginal-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_JSHAPED_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.evaluate_all \
		$(LSP_GNN_CORE_ARGS) \
		--input_type $(INPUT_TYPE) \
		--map_type jshaped \
	 	--current_seed $(seed) \
	 	--image_filename jshaped_learned_$(seed).png \
		--save_dir /data/$(LSP_GNN_JSHAPED_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt \
		--cnn_network_file /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt \
		--gcn_network_file /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt \
		--autoencoder_network_file /data/$(LSP_GNN_JSHAPED_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \

.PHONY: lsp-gnn-jshaped-results
lsp-gnn-jshaped-results:
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.plotting \
		--data_file /data/$(LSP_GNN_JSHAPED_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_GNN_JSHAPED_BASENAME)/results/jshaped_results_$(EXPERIMENT_NAME).png \
		--base_resolution 0.4


#---=== Parallel Hallway Environment Experiments ===---#
LSP_GNN_PH_BASENAME ?= lsp_gnn/parallel
LSP_GNN_PH_NUM_TRAINING_SEEDS ?= 1000
LSP_GNN_PH_NUM_TESTING_SEEDS ?= 500
LSP_GNN_PH_NUM_EVAL_SEEDS ?= 500

lsp-gnn-ph-data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(LSP_GNN_PH_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/data_completion_logs/data_training_$${ii}.txt"; done) \
	$(shell for ii in $$(seq 30000 $$((30000 + $(LSP_GNN_PH_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/data_completion_logs/data_testing_$${ii}.txt"; done)

$(lsp-gnn-ph-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-gnn-ph-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-gnn-ph-data-gen-seeds):
	@echo "Generating Data [$(LSP_GNN_PH_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/data_completion_logs
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/error_logs
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.gen_data \
		$(LSP_GNN_CORE_ARGS) \
		--map_type new_office \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(LSP_GNN_PH_BASENAME)/

lsp-gnn-ph-autoencoder-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt
$(lsp-gnn-ph-autoencoder-train-file): $(lsp-gnn-ph-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 28000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 7000 \
		--save_dir /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_PH_BASENAME)/ 

lsp-gnn-ph-cnn-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt
$(lsp-gnn-ph-cnn-train-file): $(lsp-gnn-ph-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_cnn_lsp \
		--save_dir /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_PH_BASENAME)/ 

lsp-gnn-ph-marginal-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt 
$(lsp-gnn-ph-marginal-train-file): $(lsp-gnn-ph-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_marginal_lsp \
		--save_dir /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_PH_BASENAME)/ 

lsp-gnn-ph-eval-seeds = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_GNN_PH_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/results/$(EXPERIMENT_NAME)/parallel_learned_$${ii}.png"; done)
$(lsp-gnn-ph-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-gnn-ph-eval-seeds): $(lsp-gnn-ph-cnn-train-file)
$(lsp-gnn-ph-eval-seeds): $(lsp-gnn-ph-marginal-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_PH_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.evaluate_all \
		$(LSP_GNN_CORE_ARGS) \
		--input_type $(INPUT_TYPE) \
		--map_type new_office \
	 	--current_seed $(seed) \
	 	--image_filename parallel_learned_$(seed).png \
		--save_dir /data/$(LSP_GNN_PH_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt \
		--cnn_network_file /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt \
		--gcn_network_file /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt \
		--autoencoder_network_file /data/$(LSP_GNN_PH_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \

.PHONY: lsp-gnn-ph-results
lsp-gnn-ph-results:
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.plotting \
		--data_file /data/$(LSP_GNN_PH_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_GNN_PH_BASENAME)/results/parallel_results_$(EXPERIMENT_NAME).png \
		--base_resolution 0.4


#---=== University Building Floorplans Experiments ===---#
LSP_GNN_FLOORPLANS_BASENAME ?= lsp_gnn/floorplans
LSP_GNN_FLOORPLANS_NUM_TRAINING_SEEDS ?= 1000
LSP_GNN_FLOORPLANS_NUM_TESTING_SEEDS ?= 500
LSP_GNN_FLOORPLANS_NUM_EVAL_SEEDS ?= 500

lsp-gnn-floorplans-data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(LSP_GNN_FLOORPLANS_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/data_completion_logs/data_training_$${ii}.txt"; done) \
	$(shell for ii in $$(seq 30000 $$((30000 + $(LSP_GNN_FLOORPLANS_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/data_completion_logs/data_testing_$${ii}.txt"; done)

$(lsp-gnn-floorplans-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-gnn-floorplans-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-gnn-floorplans-data-gen-seeds):
	@echo "Generating Data [$(LSP_GNN_FLOORPLANS_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/data_completion_logs
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/error_logs
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.gen_data \
		$(LSP_GNN_CORE_ARGS) \
		--map_type ploader \
		--base_resolution 0.1 \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/

lsp-gnn-floorplans-autoencoder-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt
$(lsp-gnn-floorplans-autoencoder-train-file): $(lsp-gnn-floorplans-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 28000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 7000 \
		--save_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/ 

lsp-gnn-floorplans-cnn-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt
$(lsp-gnn-floorplans-cnn-train-file): $(lsp-gnn-floorplans-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_cnn_lsp \
		--save_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/ 

lsp-gnn-floorplans-marginal-train-file = $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt 
$(lsp-gnn-floorplans-marginal-train-file): $(lsp-gnn-floorplans-data-gen-seeds)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.train \
		$(LSP_GNN_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_marginal_lsp \
		--save_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/ 

lsp-gnn-floorplans-eval-seeds = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_GNN_FLOORPLANS_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/results/$(EXPERIMENT_NAME)/floorplan_learned_$${ii}.png"; done)
$(lsp-gnn-floorplans-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-gnn-floorplans-eval-seeds): $(lsp-gnn-floorplans-cnn-train-file)
$(lsp-gnn-floorplans-eval-seeds): $(lsp-gnn-floorplans-marginal-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_GNN_FLOORPLANS_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.evaluate_all \
		$(LSP_GNN_CORE_ARGS) \
		--input_type $(INPUT_TYPE) \
		--map_type ploader \
		--base_resolution 0.1 \
	 	--current_seed $(seed) \
	 	--image_filename floorplan_learned_$(seed).png \
		--save_dir /data/$(LSP_GNN_FLOORPLANS_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt \
		--cnn_network_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt \
		--gcn_network_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt \
		--autoencoder_network_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \

.PHONY: lsp-gnn-floorplans-results
lsp-gnn-floorplans-results:
	@$(DOCKER_PYTHON) -m lsp_gnn.scripts.plotting \
		--data_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_GNN_FLOORPLANS_BASENAME)/results/floorplans_results_$(EXPERIMENT_NAME).png \
		--base_resolution 0.1


# ==== Helper Targets ====
.PHONY: lsp-gnn-jshaped-data-gen lsp-gnn-jshaped-train lsp-gnn-jshaped-eval lsp-gnn-jshaped-results lsp-gnn-jshaped
lsp-gnn-jshaped-data-gen: $(lsp-gnn-jshaped-data-gen-seeds)
lsp-gnn-jshaped-train: DOCKER_ARGS ?= -it
lsp-gnn-jshaped-train: $(lsp-gnn-jshaped-cnn-train-file)
lsp-gnn-jshaped-train: $(lsp-gnn-jshaped-marginal-train-file)
lsp-gnn-jshaped-eval: $(lsp-gnn-jshaped-eval-seeds)
lsp-gnn-jshaped: lsp-gnn-jshaped-eval
	$(MAKE) lsp-gnn-jshaped-results

.PHONY: lsp-gnn-ph-data-gen lsp-gnn-ph-train lsp-gnn-ph-eval lsp-gnn-ph-results lsp-gnn-ph
lsp-gnn-ph-data-gen: $(lsp-gnn-ph-data-gen-seeds)
lsp-gnn-ph-train: DOCKER_ARGS ?= -it
lsp-gnn-ph-train: $(lsp-gnn-ph-cnn-train-file)
lsp-gnn-ph-train: $(lsp-gnn-ph-marginal-train-file)
lsp-gnn-ph-eval: $(lsp-gnn-ph-eval-seeds)
lsp-gnn-ph: lsp-gnn-ph-eval
	$(MAKE) lsp-gnn-ph-results

.PHONY: lsp-gnn-floorplans-data-gen lsp-gnn-floorplans-train lsp-gnn-floorplans-eval lsp-gnn-floorplans-results lsp-gnn-floorplans
lsp-gnn-floorplans-data-gen: $(lsp-gnn-floorplans-data-gen-seeds)
lsp-gnn-floorplans-train: DOCKER_ARGS ?= -it
lsp-gnn-floorplans-train: $(lsp-gnn-floorplans-cnn-train-file)
lsp-gnn-floorplans-train: $(lsp-gnn-floorplans-marginal-train-file)
lsp-gnn-floorplans-eval: $(lsp-gnn-floorplans-eval-seeds)
lsp-gnn-floorplans: lsp-gnn-floorplans-eval
	$(MAKE) lsp-gnn-floorplans-results

.PHONY: lsp-gnn
lsp-gnn: lsp-gnn-jshaped-eval lsp-gnn-ph-eval lsp-gnn-floorplans-eval
	$(MAKE) lsp-gnn-jshaped-results
	$(MAKE) lsp-gnn-ph-results
	$(MAKE) lsp-gnn-floorplans-results
