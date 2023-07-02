
help::
	@echo "Learned Subgoal Planning (lsp):"
	@echo "  lsp-maze	Runs the 'guided maze' experiments."
	@echo ""

LSP_MAZE_BASENAME ?= lsp/maze
LSP_MAZE_NUM_TRAINING_SEEDS ?= 500
LSP_MAZE_NUM_TESTING_SEEDS ?= 100
LSP_MAZE_NUM_EVAL_SEEDS ?= 1000
LSP_MAZE_CORE_ARGS ?= --unity_path /unity/$(UNITY_BASENAME).x86_64 \
		--map_type maze \
		--save_dir /data/$(LSP_MAZE_BASENAME)/

lsp-maze-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(LSP_MAZE_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_collect_plots/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 2000 $$((2000 + $(LSP_MAZE_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_collect_plots/data_testing_$${ii}.png"; done)

$(lsp-maze-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-maze-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-maze-data-gen-seeds):
	@echo "Generating Data [$(LSP_MAZE_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_data/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_generate_data \
		$(LSP_MAZE_CORE_ARGS) \
		--save_dir /data/$(LSP_MAZE_BASENAME)/training_data/ \
	 	--current_seed $(seed) --data_file_base_name data_$(traintest)

.SECONDARY: $(lsp-maze-data-gen-seeds)

lsp-maze-train-file = $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt
$(lsp-maze-train-file): $(lsp-maze-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)
	$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_train_net \
		--save_dir /data/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_MAZE_BASENAME)/training_data/

lsp-maze-eval-seeds = \
	$(shell for ii in $$(seq 10000 $$((10000 + $(LSP_MAZE_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)/maze_learned_$${ii}.png"; done)
$(lsp-maze-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-maze-eval-seeds): $(lsp-maze-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_eval \
		$(LSP_MAZE_CORE_ARGS) \
		--save_dir /data/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt \
	 	--current_seed $(seed) --image_filename maze_learned_$(seed).png

lsp-maze-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		--data_file /data/$(LSP_MAZE_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_MAZE_BASENAME)/results/results_$(EXPERIMENT_NAME).png

# ==== Office (office2) Experiments ===

LSP_OFFICE_BASENAME ?= lsp/office
LSP_OFFICE_NUM_TRAINING_SEEDS ?= 500
LSP_OFFICE_NUM_TESTING_SEEDS ?= 100
LSP_OFFICE_NUM_EVAL_SEEDS ?= 1000
LSP_OFFICE_CORE_ARGS ?= --unity_path /unity/$(UNITY_BASENAME).x86_64 \
		--map_type office2 \
		--save_dir /data/$(LSP_OFFICE_BASENAME)/

lsp-office-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 $$((1000 + $(LSP_OFFICE_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_data/data_collect_plots/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 2000 $$((2000 + $(LSP_OFFICE_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_data/data_collect_plots/data_testing_$${ii}.png"; done)

$(lsp-office-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-office-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-office-data-gen-seeds):
	@echo "Generating Data [$(LSP_OFFICE_BASENAME) | seed: $(seed) | $(traintest)"]
	@$(call xhost_activate)
	@-rm -f $(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_data/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_data/data
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_data/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_generate_data \
		$(LSP_OFFICE_CORE_ARGS) \
		--save_dir /data/$(LSP_OFFICE_BASENAME)/training_data/ \
	 	--current_seed $(seed) --data_file_base_name data_$(traintest)

.SECONDARY: $(lsp-office-data-gen-seeds)

lsp-office-train-file = $(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt
$(lsp-office-train-file): $(lsp-office-data-gen-seeds)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME)
	$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_train_net \
		--save_dir /data/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(LSP_OFFICE_BASENAME)/training_data/

lsp-office-eval-seeds = \
	$(shell for ii in $$(seq 10000 $$((10000 + $(LSP_OFFICE_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/results/$(EXPERIMENT_NAME)/office_learned_$${ii}.png"; done)
$(lsp-office-eval-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-office-eval-seeds): $(lsp-office-train-file)
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_OFFICE_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_eval \
		$(LSP_OFFICE_CORE_ARGS) \
		--save_dir /data/$(LSP_OFFICE_BASENAME)/results/$(EXPERIMENT_NAME) \
		--network_file /data/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt \
	 	--current_seed $(seed) --image_filename office_learned_$(seed).png

lsp-office-results:
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_plotting \
		--data_file /data/$(LSP_OFFICE_BASENAME)/results/$(EXPERIMENT_NAME)/logfile.txt \
		--output_image_file /data/$(LSP_OFFICE_BASENAME)/results/results_$(EXPERIMENT_NAME).png

# ==== Helper Targets ====

.PHONY: lsp-maze-data-gen lsp-maze-train lsp-maze-eval lsp-maze-results lsp-maze
lsp-maze-data-gen: $(lsp-maze-data-gen-seeds)
lsp-maze-train: $(lsp-maze-train-file)
lsp-maze-eval: $(lsp-maze-eval-seeds)
lsp-maze: lsp-maze-eval
	$(MAKE) lsp-maze-results

.PHONY: lsp-office-data-gen lsp-office-train lsp-office-eval lsp-office-results lsp-office
lsp-office-data-gen: $(lsp-office-data-gen-seeds)
lsp-office-train: $(lsp-office-train-file)
lsp-office-eval: $(lsp-office-eval-seeds)
lsp-office: lsp-office-eval
	$(MAKE) lsp-office-results


# ==== Check that the code is functioning ====

lsp-maze-check: DATA_BASE_DIR = $(shell pwd)/data/check
lsp-maze-check: LSP_MAZE_NUM_TRAINING_SEEDS = 12
lsp-maze-check: LSP_MAZE_NUM_TESTING_SEEDS = 4
lsp-maze-check: LSP_MAZE_NUM_EVAL_SEEDS = 12
lsp-maze-check: build
	$(MAKE) lsp-maze DATA_BASE_DIR=$(DATA_BASE_DIR) \
		LSP_MAZE_NUM_TRAINING_SEEDS=$(LSP_MAZE_NUM_TRAINING_SEEDS) \
		LSP_MAZE_NUM_TESTING_SEEDS=$(LSP_MAZE_NUM_TESTING_SEEDS) \
		LSP_MAZE_NUM_EVAL_SEEDS=$(LSP_MAZE_NUM_EVAL_SEEDS)

lsp-office-check: DATA_BASE_DIR = $(shell pwd)/data/check
lsp-office-check: LSP_OFFICE_NUM_TRAINING_SEEDS = 12
lsp-office-check: LSP_OFFICE_NUM_TESTING_SEEDS = 4
lsp-office-check: LSP_OFFICE_NUM_EVAL_SEEDS = 12
lsp-office-check: build
	$(MAKE) lsp-office DATA_BASE_DIR=$(DATA_BASE_DIR) \
		LSP_OFFICE_NUM_TRAINING_SEEDS=$(LSP_OFFICE_NUM_TRAINING_SEEDS) \
		LSP_OFFICE_NUM_TESTING_SEEDS=$(LSP_OFFICE_NUM_TESTING_SEEDS) \
		LSP_OFFICE_NUM_EVAL_SEEDS=$(LSP_OFFICE_NUM_EVAL_SEEDS)
