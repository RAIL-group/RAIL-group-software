help::
	@echo "Conditional Learned Subgoal Planning (lsp-cond):"
	@echo "  lsp-cond-gen-graph-data creates the data for the experiments."
	@echo "  lsp-cond-train-autoencoder trains the auto encoder model."
	@echo "  lsp-cond-train-cnn trains the base/CNN lsp model."
	@echo "  lsp-cond-train-marginal trains the marginal GCN model."
	@echo "  lsp-cond-train trains the conditional GCN model."
	@echo "  lsp-cond-eval plans using the trained model vs baseline."
	@echo ""

LSP_WEIGHT ?= 1
LOC_WEIGHT ?= 1
LOSS ?= l1
RPW ?= 1
INPUT_TYPE ?= wall_class

LSP_COND_BASENAME ?= lsp_conditional
LSP_COND_UNITY_BASENAME ?= rail_sim_2022_07
LSP_COND_NUM_TRAINING_SEEDS ?= 2000
LSP_COND_NUM_TESTING_SEEDS ?= 500
LSP_COND_NUM_EVAL_SEEDS ?= 500

LSP_COND_CORE_ARGS ?= --save_dir /data/ \
		--unity_path /unity/$(LSP_COND_UNITY_BASENAME).x86_64 \
		--map_type new_office \
		--base_resolution 0.5 \
		--inflation_radius_m 0.75 \
		--field_of_view_deg 360 \
		--laser_max_range_m 18 \
		--pickle_directory data_pickles

LSP_COND_DATA_GEN_ARGS ?= $(LSP_COND_CORE_ARGS) \
		--save_dir /data/$(LSP_COND_BASENAME)/ 

LSP_COND_TRAINING_ARGS ?= --test_log_frequency 10 \
		--save_dir /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_COND_BASENAME)/ \
		--lsp_weight $(LSP_WEIGHT) \
		--loc_weight $(LOC_WEIGHT) \
		--loss $(LOSS) \
		--relative_positive_weight $(RPW) \
		--input_type $(INPUT_TYPE)

LSP_COND_EVAL_ARGS ?= $(LSP_COND_CORE_ARGS) \
		--save_dir /data/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt \
		--cnn_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt \
		--gcn_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt \
		--autoencoder_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--input_type $(INPUT_TYPE)

lsp-cond-data-gen-seeds = \
	$(shell for ii in $$(seq 0000 $$((0000 + $(LSP_COND_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_completion_logs/data_training_$${ii}.txt"; done) \
	$(shell for ii in $$(seq 30000 $$((30000 + $(LSP_COND_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_completion_logs/data_testing_$${ii}.txt"; done)

$(lsp-cond-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-cond-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-data-gen-seeds):
	@echo "Generating Data [seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/data_completion_logs
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/error_logs
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.gen_graph_data \
		$(LSP_COND_DATA_GEN_ARGS) \
	 	--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest)

.PHONY: lsp-cond-gen-graph-data
lsp-cond-gen-graph-data: $(lsp-cond-data-gen-seeds)

lsp-cond-autoencoder-train-file = $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt
$(lsp-cond-autoencoder-train-file): $(lsp-cond-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_train \
		$(LSP_COND_TRAINING_ARGS) \
		--num_steps 28000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 7000

lsp-cond-cnn-train-file = $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/lsp.pt
$(lsp-cond-cnn-train-file): #$(lsp-cond-autoencoder-train-file)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_train \
		$(LSP_COND_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_cnn_lsp

lsp-cond-marginal-train-file = $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/mlsp.pt 
$(lsp-cond-marginal-train-file): #$(lsp-cond-autoencoder-train-file)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_train \
		$(LSP_COND_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-3 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt \
		--train_marginal_lsp

lsp-cond-train-file = $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/model.pt 
$(lsp-cond-train-file): $(lsp-cond-autoencoder-train-file)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_train \
		$(LSP_COND_TRAINING_ARGS) \
		--num_steps 50000 \
		--learning_rate 1e-4 \
		--learning_rate_decay_factor .6 \
		--epoch_size 10000 \
		--autoencoder_network_file /data/$(LSP_COND_BASENAME)/logs/$(EXPERIMENT_NAME)/AutoEncoder.pt 

.PHONY: lsp-cond-train lsp-cond-train-autoencoder lsp-cond-train-cnn lsp-cond-train-marginal
lsp-cond-train-autoencoder: DOCKER_ARGS ?= -it
lsp-cond-train-autoencoder: $(lsp-cond-autoencoder-train-file)
lsp-cond-train-cnn: DOCKER_ARGS ?= -it
lsp-cond-train-cnn: $(lsp-cond-cnn-train-file)
lsp-cond-train-marginal: DOCKER_ARGS ?= -it
lsp-cond-train-marginal: $(lsp-cond-marginal-train-file)
lsp-cond-train: DOCKER_ARGS ?= -it
lsp-cond-train: $(lsp-cond-train-file)


LSP_COND_PLOTTING_ARGS = \
		--data_file /data/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_COND_BASENAME)/results/results_$(EXPERIMENT_NAME).png

.PHONY: lsp-cond-results
lsp-cond-results:
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.cond_lsp_plotting \
		$(LSP_COND_PLOTTING_ARGS)
.PHONY: lsp-cond-results-marginal
lsp-cond-results-marginal:
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.cond_lsp_plotting \
		--data_file /data/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/mlsp_logfile.txt \
		--gnn

lsp-cond-eval-seeds = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_COND_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/maze_learned_$${ii}.png"; done)
$(lsp-cond-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-eval-seeds): $(lsp-cond-cnn-train-file)
$(lsp-cond-eval-seeds): $(lsp-cond-marginal-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_evaluate \
		$(LSP_COND_EVAL_ARGS) \
	 	--current_seed $(seed) \
	 	--image_filename maze_learned_$(seed).png

.PHONY: lsp-cond-eval
lsp-cond-eval: $(lsp-cond-eval-seeds)
	$(MAKE) lsp-cond-results

lsp-cond-eval-seeds-base = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_COND_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/maze_lsp_$${ii}.png"; done)
$(lsp-cond-eval-seeds-base): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-eval-seeds-base): $(lsp-cond-cnn-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_eval_gcn \
		$(LSP_COND_EVAL_ARGS) \
	 	--current_seed $(seed) \
	 	--image_filename maze_lsp_$(seed).png \
	 	--logfile_name lsp_logfile.txt

.PHONY: lsp-cond-eval-base
lsp-cond-eval-base: $(lsp-cond-eval-seeds-base)

lsp-cond-eval-seeds-marginal = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_COND_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/maze_mlsp_$${ii}.png"; done)
$(lsp-cond-eval-seeds-marginal): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-eval-seeds-marginal): $(lsp-cond-marginal-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_eval_gcn \
		$(LSP_COND_EVAL_ARGS) \
	 	--current_seed $(seed) \
	 	--image_filename maze_mlsp_$(seed).png \
	 	--logfile_name mlsp_logfile.txt

.PHONY: lsp-cond-eval-marginal
lsp-cond-eval-marginal: $(lsp-cond-eval-seeds-marginal)
	$(MAKE) lsp-cond-results-marginal

lsp-cond-eval-seeds-conditional = \
	$(shell for ii in $$(seq 50000 $$((50000 + $(LSP_COND_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)/maze_clsp_$${ii}.png"; done)
$(lsp-cond-eval-seeds-conditional): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-cond-eval-seeds-conditional): $(lsp-cond-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_COND_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m lsp_cond.scripts.lsp_cond_eval_gcn \
		$(LSP_COND_EVAL_ARGS) \
	 	--current_seed $(seed) \
	 	--image_filename maze_clsp_$(seed).png \
	 	--logfile_name clsp_logfile.txt

.PHONY: lsp-cond-eval-conditional
lsp-cond-eval-conditional: $(lsp-cond-eval-seeds-conditional)
