MRLSP_BASENAME = mrlsp
MRLSP_UNITY_BASENAME ?= $(UNITY_BASENAME)

MRLSP_CORE_ARGS ?= --unity_path /unity/$(MRLSP_UNITY_BASENAME).x86_64 \
		--save_dir /data/$(MRLSP_BASENAME)/$(EXPERIMENT_NAME) \
		--limit_frontiers 7 \
		--iterations 40000 \

MRLSP_SEED_START = 2000
MRLSP_NUM_EXPERIMENTS = 100
define mrlsp_get_seeds
	$(shell seq $(MRLSP_SEED_START) $$(($(MRLSP_SEED_START)+$(MRLSP_NUM_EXPERIMENTS) - 1)))
endef

MRLSP_PLANNER = optimistic baseline mrlsp
MRLSP_NUM_ROBOTS = 1 2 3
all-targets = $(foreach planner, $(MRLSP_PLANNER), \
	$(foreach num_robots, $(MRLSP_NUM_ROBOTS), \
		$(foreach seed, $(call mrlsp_get_seeds), \
			$(DATA_BASE_DIR)/$(MRLSP_BASENAME)/$(EXPERIMENT_NAME)/planner_$(planner)_$(ENV)_cost_$(seed)_r$(num_robots).txt)))

.PHONY: run-mrlsp
run-mrlsp: $(all-targets)
$(all-targets): num_robots = $(shell echo $@ | grep -oE 'r[0-9]+' | grep -oE '[0-9]+')
$(all-targets): planner = $(shell echo $@ | grep -oE 'planner_[a-z]+' | cut -d'_' -f2)
$(all-targets): seed = $(shell echo $@ | grep -oE 'cost_[0-9]+' | cut -d'_' -f2) 
$(all-targets): map_type = $(ENV)
$(all-targets): network_file = $(NETWORK_FILE)
$(all-targets):
	@echo $(network_file)
	$(call xhost_activate)
	@echo "Current experiment: Planner = $(planner) | Map = $(map_type) | Seed = $(seed) | Num robots = $(num_robots)"
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_BASENAME)/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m modules.mrlsp.mrlsp.scripts.simulation_$(planner) \
		$(MRLSP_CORE_ARGS) \
		--seed $(seed) \
		--num_robots $(num_robots) \
		--map_type $(map_type) \
		--network_file $(network_file)

.PHONY: mrlsp-maze
mrlsp-maze: $(lsp-maze-train-file)
mrlsp-maze:
	@$(MAKE) run-mrlsp ENV=maze \
		NETWORK_FILE=/data/$(LSP_MAZE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt

.PHONY: mrlsp-office
mrlsp-office: $(lsp-office-train-file)
mrlsp-office:
	@$(MAKE) run-mrlsp ENV=office2 \
		NETWORK_FILE=/data/$(LSP_OFFICE_BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt

# Visualize different planner for different seed and number of robots in office environment
.PHONY: visualize
visualize: DOCKER_ARGS ?= -it
visualize: xhost-activate arg-check-data
	@mkdir -p $(DATA_BASE_DIR)/$(MRLSP_BASENAME)/visualize/
	@$(DOCKER_PYTHON) -m modules.mrlsp.mrlsp.scripts.simulation_$(PLANNER) \
		$(MRLSP_CORE_ARGS) \
	   	--save_dir data/$(MRLSP_BASENAME)/visualize/ \
		--network_file /data/$(BASENAME)/training_logs/$(EXPERIMENT_NAME)/VisLSPOriented.pt \
		--seed $(SEED) \
		--map_type $(MAP) \
		--do_plot True \
		--num_robots $(NUM_ROBOTS) \

.PHONY: visualize-office
visualize-office: PLANNER = optimistic
visualize-office: SEED = 2001
visualize-office: NUM_ROBOTS = 2
visualize-office: 
	@$(MAKE) visualize BASENAME=$(LSP_OFFICE_BASENAME) \
		PLANNER=$(PLANNER) \
		SEED=$(SEED) \
		NUM_ROBOTS=$(NUM_ROBOTS) \
		XPASSTHROUGH=true \
		MAP=office2

.PHONY: visualize-maze
visualize-maze: PLANNER = optimistic
visualize-maze: SEED = 2001
visualize-maze: NUM_ROBOTS = 2
visualize-maze: 
	@$(MAKE) visualize BASENAME=$(LSP_MAZE_BASENAME) \
		PLANNER=$(PLANNER) \
		SEED=$(SEED) \
		NUM_ROBOTS=$(NUM_ROBOTS) \
		XPASSTHROUGH=true \
		MAP=maze
