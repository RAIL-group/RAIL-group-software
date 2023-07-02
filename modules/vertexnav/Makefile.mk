

# This target is to make an image by calling a script within 'example'
VERTEXNAV_UNITY_BASENAME ?= $(UNITY_BASENAME)
VERTEXNAV_CORE_ARGS = \
	--unity_path /unity/$(VERTEXNAV_UNITY_BASENAME).x86_64 \
	--xpassthrough $(XPASSTHROUGH) \
	--max_range 100 \
	--num_range 32 \
	--num_bearing 128
vertexnav_dungeon_base_dir = $(DATA_BASE_DIR)/vertexnav/dungeon

vertexnav-dungeon-data-gen-seeds = \
	$(shell for ii in $$(seq 1000 1249); do echo "$(vertexnav_dungeon_base_dir)/data/training_env_plots/training_env_$$ii.png"; done) \
	$(shell for ii in $$(seq 2000 2024); do echo "$(vertexnav_dungeon_base_dir)/data/training_env_plots/testing_env_$$ii.png"; done)

$(vertexnav-dungeon-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(vertexnav-dungeon-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(vertexnav-dungeon-data-gen-seeds):
	@$(call arg_check_unity)
	@$(call arg_check_data)
	@mkdir -p $(vertexnav_dungeon_base_dir)/data/pickles/
	@mkdir -p $(vertexnav_dungeon_base_dir)/data/training_env_plots/
	@rm -f $(vertexnav_dungeon_base_dir)/data/dungeon_*_$(seed).csv
	@echo "dungeon_$(traintest)"
	@$(DOCKER_PYTHON) -m vertexnav.scripts.gen_vertex_training_data_sim \
		$(VERTEXNAV_CORE_ARGS) \
		--base_data_path /data/vertexnav/dungeon \
		--environment dungeon \
		--data_file_base_name dungeon_$(traintest) \
		--data_plot_name $(traintest)_env \
		--seed $(seed)

vertexnav-dungeon-train-net = $(vertexnav_dungeon_base_dir)/training_logs/$(EXPERIMENT_NAME)/VertexNavGrid.pt
$(vertexnav-dungeon-train-net): $(vertexnav-dungeon-data-gen-seeds)
	@mkdir -p $(vertexnav_dungeon_base_dir)/training_logs/$(EXPERIMENT_NAME)/
	@$(call arg_check_data)
	@$(DOCKER_PYTHON) -m vertexnav.scripts.train_vertex_nav_net \
	 	--training_data_file /data/vertexnav/dungeon/data/*train*.csv \
	 	--test_data_file /data/vertexnav/dungeon/data/*test*.csv \
	 	--logdir /data/vertexnav/dungeon/training_logs/$(EXPERIMENT_NAME)/ \
	 	--mini_summary_frequency 100 --summary_frequency 1000 \
	 	--num_epochs 8 \
	 	--learning_rate 0.001 \

# Dungeon Environment Parameters
DUNGEON_EXTENSION_NAME ?= dungeon
DUNGEON_UNITY_BASENAME ?= $(UNITY_BASENAME)
DUNGEON_MAX_RANGE ?= 100
DUNGEON_NUM_RANGE ?= 32
DUNGEON_NUM_BEARING ?= 128

vertexnav-dungeon-eval-seeds = \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r1_merge_none.png"; done) \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r1_merge_multi.png"; done) \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r1_merge_single.png"; done) \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r3_merge_none.png"; done) \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r3_merge_multi.png"; done) \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r3_merge_single.png"; done)

vertexnav-dungeon-eval-seeds = \
	$(shell for ii in $$(seq 10000 10099); do echo "$(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)/dungeon_slam_s$${ii}_r3_merge_none.png"; done) \

$(vertexnav-dungeon-eval-seeds): seed = $(shell echo '$@' | grep -Eo '[0-9]+' | tail -2 | head -1)
$(vertexnav-dungeon-eval-seeds): num_robots = $(shell echo '$@' | grep -Eo '[0-9]+' | tail -1)
$(vertexnav-dungeon-eval-seeds): merge_type = $(shell echo $@ | grep -Eo 'merge_(none|single|multi)' | tail -1)
$(vertexnav-dungeon-eval-seeds): $(vertexnav-dungeon-train-net)
	@echo "Random Seed: $(seed)"
	@echo "Number of robots: $(num_robots)"
	@mkdir -p $(vertexnav_dungeon_base_dir)/results/$(EXPERIMENT_NAME)
	@- $(DOCKER_PYTHON) -m vertexnav.scripts.eval_simulated \
		--video_path /data/vertexnav/dungeon/results/$(EXPERIMENT_NAME)/dungeon_slam_s$(seed)_r$(num_robots)_$(merge_type).mp4 \
		--figure_path /data/vertexnav/dungeon/results/$(EXPERIMENT_NAME)/dungeon_slam_s$(seed)_r$(num_robots)_$(merge_type).png \
		--network_file /data/vertexnav/dungeon/training_logs/$(EXPERIMENT_NAME)/VertexNavGrid.pt \
		--unity_exe_path /unity/$(DUNGEON_UNITY_BASENAME).x86_64 \
		--environment dungeon \
		--seed $(seed) \
		--do_use_robot \
		--do_explore \
		--do_use_frontiers \
		--num_robots $(num_robots) \
		--max_range $(DUNGEON_MAX_RANGE) \
		--num_range $(DUNGEON_NUM_RANGE) \
		--num_bearing $(DUNGEON_NUM_BEARING) \
		--merge_type $(merge_type) \
		--sig_r 5.0 \
		--sig_th 0.125 \
		--nn_peak_thresh 0.5
	@sleep 10

# Convenience Targets
.PHONY: vertexnav-dungeon-generate-data vertexnav-dungeon-train vertexnav-dungeon-eval vertexnav-dungeon
vertexnav-dungeon-generate-data: $(vertexnav-dungeon-data-gen-seeds)
vertexnav-dungeon-train: $(vertexnav-dungeon-train-net)
vertexnav-dungeon-eval: $(vertexnav-dungeon-eval-seeds)
vertexnav-dungeon: vertexnav-dungeon-eval
