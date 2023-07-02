
help::
	@echo "Policy Selection Experiments:"
	@echo "  lsp-policy-selection-maze		Runs the policy selection maze experiments."
	@echo "  lsp-policy-selection-office	Runs the policy selection office experiments."
	@echo "  lsp-policy-selection-check		Runs minimal policy selection maze experiments to verify everything works."
	@echo ""

LSP_SELECT_ENVIRONMENT_NAME ?= mazeA
LSP_SELECT_BASENAME ?= lsp_select
LSP_SELECT_UNITY_BASENAME ?= $(RAIL_SIM_BASENAME)
LSP_SELECT_MAP_TYPE ?= maze

LSP_SELECT_NUM_TRAINING_SEEDS ?= 500
LSP_SELECT_NUM_TESTING_SEEDS ?= 100
LSP_SELECT_CORE_ARGS ?= --unity_path /unity/$(LSP_SELECT_UNITY_BASENAME).x86_64 \
		--map_type $(LSP_SELECT_MAP_TYPE) \
		--inflation_radius_m 0.75

# Data Generation and Training
lsp-select-gen-data-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(LSP_SELECT_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME)/data_collect_plots/$(LSP_SELECT_ENVIRONMENT_NAME)_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 500 $$((500 + $(LSP_SELECT_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME)/data_collect_plots/$(LSP_SELECT_ENVIRONMENT_NAME)_testing_$${ii}.png"; done)

lsp-select-gen-data: $(lsp-select-gen-data-seeds)
$(lsp-select-gen-data-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-select-gen-data-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-select-gen-data-seeds):
	$(call xhost_activate)
	@echo "Generating Data [$(LSP_SELECT_BASENAME) | $(LSP_SELECT_ENVIRONMENT_NAME) | seed: $(seed) | $(traintest)]"
	@-rm -f $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME)_$(traintest)_$(seed).csv
	@-mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)
	@-mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME)/data
	@-mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME)/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_vis_gen_data \
		$(LSP_SELECT_CORE_ARGS) \
		--current_seed $(seed) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME) \
		--data_file_base_name $(LSP_SELECT_ENVIRONMENT_NAME)_$(traintest)

lsp-select-train-file = $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_logs/$(LSP_SELECT_ENVIRONMENT_NAME)/VisLSPOriented.pt
lsp-select-train: $(lsp-select-train-file)
$(lsp-select-train-file): $(lsp-select-gen-data-seeds)
$(lsp-select-train-file):
	@echo "Training [$(LSP_SELECT_BASENAME) | $(LSP_SELECT_ENVIRONMENT_NAME)]"
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_logs/$(LSP_SELECT_ENVIRONMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp.scripts.vis_lsp_train_net \
		--save_dir /data/$(LSP_SELECT_BASENAME)/training_logs/$(LSP_SELECT_ENVIRONMENT_NAME) \
		--data_csv_dir /data/$(LSP_SELECT_BASENAME)/training_data/$(LSP_SELECT_ENVIRONMENT_NAME)

GREENFLOOR_SIM_ID ?= greenfloor
WALLSWAP_SIM_ID ?= wallswap

LSP_SELECT_MAZE_A_ARGS ?= LSP_SELECT_ENVIRONMENT_NAME=mazeA LSP_SELECT_MAP_TYPE=maze
LSP_SELECT_MAZE_B_ARGS ?= LSP_SELECT_ENVIRONMENT_NAME=mazeB LSP_SELECT_MAP_TYPE=maze LSP_SELECT_UNITY_BASENAME=$(RAIL_SIM_BASENAME)_$(GREENFLOOR_SIM_ID)
LSP_SELECT_MAZE_C_ARGS ?= LSP_SELECT_ENVIRONMENT_NAME=mazeC LSP_SELECT_MAP_TYPE=maze

LSP_SELECT_OFFICE_BASE_ARGS ?= LSP_SELECT_ENVIRONMENT_NAME=office LSP_SELECT_MAP_TYPE=office2
LSP_SELECT_OFFICE_WALLSWAP_ARGS ?= LSP_SELECT_ENVIRONMENT_NAME=officewall LSP_SELECT_MAP_TYPE=office2 LSP_SELECT_UNITY_BASENAME=$(RAIL_SIM_BASENAME)_$(WALLSWAP_SIM_ID)


unity-simulator-additional-files = \
	$(RAIL_SIM_DIR)/v$(RAIL_SIM_VERSION)/$(RAIL_SIM_BASENAME)_$(GREENFLOOR_SIM_ID).x86_64 \
	$(RAIL_SIM_DIR)/v$(RAIL_SIM_VERSION)/$(RAIL_SIM_BASENAME)_$(WALLSWAP_SIM_ID).x86_64

.PHONY: unity-simulator-additional-data
unity-simulator-additional-data: $(unity-simulator-additional-files)
$(unity-simulator-additional-files): sim_identifier = $(shell echo $@ | grep -oP '(?<=_)[A-Za-z]+(?=\.x86_64)')
$(unity-simulator-additional-files):
	@echo "Downloading the Unity simulator data for $(sim_identifier)"
	cd $(RAIL_SIM_DIR)/v$(RAIL_SIM_VERSION) \
		&& curl -OL https://github.com/RAIL-group/RAIL-group-simulator/releases/download/v$(RAIL_SIM_VERSION)/data_$(sim_identifier).zip \
		&& unzip data_$(sim_identifier).zip \
		&& rm data_$(sim_identifier).zip
	@echo "Unity simulator data for $(sim_identifier) downloaded and unpacked."


## Maze 
.PHONY: lsp-select-gen-data-mazeA lsp-select-gen-data-mazeB lsp-select-gen-data-mazeC
lsp-select-gen-data-mazeA:
	@$(MAKE) lsp-select-gen-data $(LSP_SELECT_MAZE_A_ARGS)
lsp-select-gen-data-mazeB: unity-simulator-additional-data
lsp-select-gen-data-mazeB:
	@$(MAKE) lsp-select-gen-data $(LSP_SELECT_MAZE_B_ARGS)
lsp-select-gen-data-mazeC:
	@$(MAKE) lsp-select-gen-data $(LSP_SELECT_MAZE_C_ARGS)

.PHONY: lsp-select-train-mazeA lsp-select-train-mazeB lsp-select-train-mazeC
lsp-select-train-mazeA: lsp-select-gen-data-mazeA
lsp-select-train-mazeA:
	@$(MAKE) lsp-select-train $(LSP_SELECT_MAZE_A_ARGS)
lsp-select-train-mazeB: lsp-select-gen-data-mazeB
lsp-select-train-mazeB:
	@$(MAKE) lsp-select-train $(LSP_SELECT_MAZE_B_ARGS)
lsp-select-train-mazeC: lsp-select-gen-data-mazeB
lsp-select-train-mazeC:
	@$(MAKE) lsp-select-train $(LSP_SELECT_MAZE_C_ARGS)

## Office
.PHONY: lsp-select-gen-data-officebase lsp-select-gen-data-officewall
lsp-select-gen-data-officebase:
	@$(MAKE) lsp-select-gen-data $(LSP_SELECT_OFFICE_BASE_ARGS)
lsp-select-gen-data-officewall: unity-simulator-additional-data
lsp-select-gen-data-officewall:
	@$(MAKE) lsp-select-gen-data $(LSP_SELECT_OFFICE_WALLSWAP_ARGS)

.PHONY: lsp-select-train-officebase lsp-select-train-officewall
lsp-select-train-officebase: lsp-select-gen-data-officebase
lsp-select-train-officebase:
	@$(MAKE) lsp-select-train $(LSP_SELECT_OFFICE_BASE_ARGS)
lsp-select-train-officewall: lsp-select-gen-data-officewall
lsp-select-train-officewall:
	@$(MAKE) lsp-select-train $(LSP_SELECT_OFFICE_WALLSWAP_ARGS)


# Policy Selection
LSP_SELECT_NUM_SEEDS_DEPLOY ?= 150

MAZE_POLICIES ?= "nonlearned lspA lspB lspC"
MAZE_ENVS ?= "envA envB envC"

OFFICE_POLICIES ?= "nonlearned lspmaze lspoffice lspofficewallswap"
OFFICE_ENVS ?= "mazeA office officewall"

# Note: *_start seed variable names are used in get_seed function
envA_start_seed ?= 2000
envB_start_seed ?= 3000 
envC_start_seed ?= 4000
maze_costs_save_dir ?= policy_selection/maze_costs

mazeA_start_seed ?= 2000
office_start_seed ?= 3000 
officewall_start_seed ?= 4000
office_costs_save_dir ?= policy_selection/office_costs

define get_seeds
	$(eval start := $(1)_start_seed)
	$(shell seq $(value $(start)) $$(($(value $(start))+$(2)-1)))
endef

offline-replay-seeds = $(foreach policy,$(POLICIES_TO_RUN), \
								$(foreach env,$(ENVS_TO_DEPLOY), \
									$(foreach seed,$(call get_seeds, $(env), $(LSP_SELECT_NUM_SEEDS_DEPLOY)), \
										$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(replay_costs_save_dir)/target_plcy_$(policy)_envrnmnt_$(env)_$(seed).txt)))

lsp-select-offline-replay-costs: $(offline-replay-seeds)
$(offline-replay-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(offline-replay-seeds): policy = $(shell echo $@ | grep -oE 'plcy_[Aa-Zz]+' | cut -d'_' -f2)
$(offline-replay-seeds): env = $(shell echo $@ | grep -oE 'envrnmnt_[Aa-Zz]+' | cut -d'_' -f2)
$(offline-replay-seeds): sim_name = $(if $(filter $(env),envB),_$(GREENFLOOR_SIM_ID),$(if $(filter $(env),officewall),_$(WALLSWAP_SIM_ID)))
$(offline-replay-seeds): LSP_SELECT_MAP_TYPE = $(if $(or $(filter $(env),office),$(filter $(env),officewall)),office2,maze)
$(offline-replay-seeds): LSP_SELECT_UNITY_BASENAME = $(RAIL_SIM_BASENAME)$(sim_name)
$(offline-replay-seeds):
	$(call xhost_activate)
	@echo "Deploying with Offline Replay [$(policy) | $(env) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(replay_costs_save_dir)
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_offline_replay_costs \
		$(LSP_SELECT_CORE_ARGS) \
		--experiment_type $(EXPERIMENT_TYPE) \
		--seed $(seed) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/$(replay_costs_save_dir) \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner $(policy) \
		--env $(env)

.PHONY: lsp-select-offline-replay-costs-maze
lsp-select-offline-replay-costs-maze: lsp-select-train-mazeA lsp-select-train-mazeB lsp-select-train-mazeC
lsp-select-offline-replay-costs-maze:
	@$(MAKE) lsp-select-offline-replay-costs \
		EXPERIMENT_TYPE=maze \
		POLICIES_TO_RUN=$(MAZE_POLICIES) \
		ENVS_TO_DEPLOY=$(MAZE_ENVS) \
		replay_costs_save_dir=$(maze_costs_save_dir)

.PHONY: lsp-select-offline-replay-costs-office
lsp-select-offline-replay-costs-office: lsp-select-train-mazeA lsp-select-train-officebase lsp-select-train-officewall
lsp-select-offline-replay-costs-office:
	@$(MAKE) lsp-select-offline-replay-costs \
		EXPERIMENT_TYPE=office \
		POLICIES_TO_RUN=$(OFFICE_POLICIES) \
		ENVS_TO_DEPLOY=$(OFFICE_ENVS) \
		replay_costs_save_dir=$(office_costs_save_dir)

.PHONY: lsp-policy-selection-maze lsp-policy-selection-office
lsp-policy-selection-maze: lsp-select-offline-replay-costs-maze
lsp-policy-selection-maze: DOCKER_ARGS ?= -it
lsp-policy-selection-maze: xhost-activate
lsp-policy-selection-maze:
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_policy_selection_results \
		--save_dir /data/$(LSP_SELECT_BASENAME)/$(maze_costs_save_dir) \
		--experiment_type maze \
		--start_seeds $(envA_start_seed) $(envB_start_seed) $(envC_start_seed) \
		--num_seeds $(LSP_SELECT_NUM_SEEDS_DEPLOY)

lsp-policy-selection-office: lsp-select-offline-replay-costs-office
lsp-policy-selection-office: DOCKER_ARGS ?= -it
lsp-policy-selection-office: xhost-activate
lsp-policy-selection-office:
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_policy_selection_results \
		--save_dir /data/$(LSP_SELECT_BASENAME)/$(office_costs_save_dir) \
		--experiment_type office
		--start_seeds $(mazeA_start_seed) $(office_start_seed) $(officewall_start_seed) \
		--num_seeds $(LSP_SELECT_NUM_SEEDS_DEPLOY)

.PHONY: offline-replay-demo
offline-replay-demo: DOCKER_ARGS ?= -it
offline-replay-demo: xhost-activate
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/offline-replay-demo
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_offline_replay_demo \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 3087 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/offline-replay-demo \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner lspoffice \
		--env office \
		--do_plot

.PHONY: lsp-policy-selection-check
lsp-policy-selection-check:
	@$(MAKE) lsp-policy-selection-maze \
		LSP_SELECT_NUM_TRAINING_SEEDS=1 \
		LSP_SELECT_NUM_TESTING_SEEDS=1 \
		LSP_SELECT_BASENAME=lsp_select_check \
		LSP_SELECT_NUM_SEEDS_DEPLOY=2
