
help::
	@echo "Learned Subgoal Planning (lsp):"
	@echo "  lsp-maze	Runs the 'guided maze' experiments."
	@echo ""

EXPERIMENT_NAME = lsp_select
ENVIRONMENT_NAME = maze
LSP_SELECT_BASENAME = lsp_select
LSP_SELECT_UNITY_BASENAME ?= rail_sim_2022_07

LSP_SELECT_CORE_ARGS ?= --unity_path /unity/$(LSP_SELECT_UNITY_BASENAME).x86_64 \
		--step_size 1.8 \
		--field_of_view_deg 360 \
		--map_type office2 \
		--base_resolution 0.5 \
		--inflation_radius_m 0.75 \
		--laser_max_range_m 18

lsp-select-gen-data-seeds = \
	$(shell for ii in $$(seq 0 499); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(ENVIRONMENT_NAME)/data_collect_plots/$(ENVIRONMENT_NAME)_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 500 599); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(ENVIRONMENT_NAME)/data_collect_plots/$(ENVIRONMENT_NAME)_testing_$${ii}.png"; done)

.PHONY: lsp-select-gen-data
lsp-select-gen-data: $(lsp-select-gen-data-seeds)
# $(lsp-select-gen-data-seeds): DOCKER_ARGS ?= -it
$(lsp-select-gen-data-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(lsp-select-gen-data-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-select-gen-data-seeds):
	$(call xhost_activate)
	@echo "Generating Data [$(LSP_SELECT_BASENAME) | seed: $(seed) | $(traintest)]"
	@-rm -f $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(ENVIRONMENT_NAME)_$(traintest)_$(seed).csv
	@-mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)
	@-mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(ENVIRONMENT_NAME)/data
	@-mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_data/$(ENVIRONMENT_NAME)/data_collect_plots
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_vis_gen_data \
		$(LSP_SELECT_CORE_ARGS) \
		--current_seed $(seed) \
		--save_dir /data/lsp_select/training_data/$(ENVIRONMENT_NAME) \
		--data_file_base_name $(ENVIRONMENT_NAME)_$(traintest)

.PHONY: lsp-select-gen-data-mazeA
lsp-select-gen-data-mazeA: ENVIRONMENT_NAME = mazeA
lsp-select-gen-data-mazeA: $(lsp-select-gen-data-seeds)

.PHONY: lsp-select-gen-data-mazeB
lsp-select-gen-data-mazeB: ENVIRONMENT_NAME = mazeB
lsp-select-gen-data-mazeB: LSP_SELECT_UNITY_BASENAME = rail_sim_2022_07_greenfloor
lsp-select-gen-data-mazeB: $(lsp-select-gen-data-seeds)

.PHONY: lsp-select-gen-data-mazeC
lsp-select-gen-data-mazeC: ENVIRONMENT_NAME = mazeC
lsp-select-gen-data-mazeC: $(lsp-select-gen-data-seeds)


lsp-select-train-file = $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_logs/$(ENVIRONMENT_NAME)/VisLSPOriented.pt
$(lsp-select-train-file): $(lsp-select-gen-data-seeds)
$(lsp-select-train-file): DOCKER_ARGS ?= -it
$(lsp-select-train-file):
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/training_logs/$(ENVIRONMENT_NAME)
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_vis_train \
		--save_dir /data/$(LSP_SELECT_BASENAME)/training_logs/$(ENVIRONMENT_NAME) \
		--data_csv_dir /data/$(LSP_SELECT_BASENAME)/training_data/$(ENVIRONMENT_NAME) \
		--num_epochs 11 \
		--learning_rate 0.003 \
		--id 42

.PHONY: lsp-select-train
lsp-select-train: $(lsp-select-train-file)

.PHONY: lsp-select-train-mazeA
lsp-select-train-mazeA: ENVIRONMENT_NAME = mazeA
lsp-select-train-mazeA: $(lsp-select-train-file)

.PHONY: lsp-select-train-mazeB
lsp-select-train-mazeB: ENVIRONMENT_NAME = mazeB
lsp-select-train-mazeB: LSP_SELECT_UNITY_BASENAME = rail_sim_2022_07_greenfloor
lsp-select-train-mazeB: $(lsp-select-train-file)

.PHONY: lsp-select-train-mazeC
lsp-select-train-mazeC: ENVIRONMENT_NAME = mazeC
lsp-select-train-mazeC: $(lsp-select-train-file)


.PHONY: lsp-cost-dist
lsp-cost-dist: DOCKER_ARGS ?= -it
lsp-cost-dist: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/nav_cost_data
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_cost_dist \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 0 300 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/nav_cost_data \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/VisLSPOriented_std2.pt

.PHONY: lsp-model-selection
lsp-model-selection: DOCKER_ARGS ?= -it
lsp-model-selection: xhost-activate arg-check-data arg-check-unity
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_model_selection \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 0 1000 \
		--save_dir /data/$(LSP_SELECT_BASENAME) \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/VisLSPOriented_std2.pt \
		--corrupt_pose_prob 1.0

.PHONY: lsp-gen-nav-cost-data
lsp-gen-nav-cost-data: DOCKER_ARGS ?= -it
lsp-gen-nav-cost-data: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/nav_cost_data
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/nav_cost_data/data
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_gen_nav_cost_data \
		$(LSP_SELECT_CORE_ARGS) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/nav_cost_data \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/VisLSPOriented_std2.pt \
		--seed 702 \
		--no_of_maps 250

.PHONY: lsp-model-selection-eval
lsp-model-selection-eval: DOCKER_ARGS ?= -it
lsp-model-selection-eval: xhost-activate arg-check-data arg-check-unity
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_model_selection_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/nav_cost_data \
		--stepwise_eval

.PHONY: cyclegan-data-gen-A
cyclegan-data-gen-A: DOCKER_ARGS ?= -it
cyclegan-data-gen-A: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeA2A/trainB
	@$(DOCKER_PYTHON) -m lsp_select.scripts.cyclegan_data_gen \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 10 23 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeA2A/trainB

.PHONY: cyclegan-data-gen-B
cyclegan-data-gen-B: LSP_SELECT_UNITY_BASENAME = rail_sim_urp_2022B
cyclegan-data-gen-B: DOCKER_ARGS ?= -it
cyclegan-data-gen-B: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeA2B/trainB
	@$(DOCKER_PYTHON) -m lsp_select.scripts.cyclegan_data_gen \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 10 23  \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeA2B/trainB

.PHONY: cyclegan-data-gen-C
cyclegan-data-gen-C: DOCKER_ARGS ?= -it
cyclegan-data-gen-C: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeA2C/trainB
	@$(DOCKER_PYTHON) -m lsp_select.scripts.cyclegan_data_gen \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 10 20 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeA2C/trainB \
		--envC

.PHONY: lsp-gen-threshold
lsp-gen-threshold: DOCKER_ARGS ?= -it
lsp-gen-threshold: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/subgoal_labels_data/EnvA
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_gen_threshold \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 0 100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/subgoal_labels_data/EnvA \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/EnvA.pt

.PHONY: lsp-rtm-eval-A
lsp-rtm-eval-A: DOCKER_ARGS ?= -it
lsp-rtm-eval-A: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/planner_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_rtm_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 2000 2100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/planner_eval \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/EnvA.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2A/latest_net_G_B.pth \
		--env envA \
		--threshold 1.8

.PHONY: lsp-rtm-eval-B
lsp-rtm-eval-B: DOCKER_ARGS ?= -it
lsp-rtm-eval-B: LSP_SELECT_UNITY_BASENAME = rail_sim_urp_2022B
lsp-rtm-eval-B: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/planner_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_rtm_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 3000 3100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/planner_eval \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/EnvA.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2B/latest_net_G_B.pth \
		--env envB \
		--threshold 1.8

.PHONY: lsp-rtm-eval-C
lsp-rtm-eval-C: DOCKER_ARGS ?= -it
lsp-rtm-eval-C: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/planner_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_rtm_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 4000 4100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/planner_eval \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/EnvA.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2C/latest_net_G_B.pth \
		--env envC \
		--threshold 1.8

.PHONY: lsp-planner-eval-A
lsp-planner-eval-A: DOCKER_ARGS ?= -it
lsp-planner-eval-A: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeABC_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_planner_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 2000 2100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/mazeA/mazeA.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2A/latest_net_G_B.pth \
		--planner nonlearned \
		--env envA

.PHONY: lsp-planner-eval-B
lsp-planner-eval-B: LSP_SELECT_UNITY_BASENAME = rail_sim_2022_07_greenfloor
lsp-planner-eval-B: DOCKER_ARGS ?= -it
lsp-planner-eval-B: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeABC_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_planner_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 3000 3100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/mazeA/mazeA.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2B/latest_net_G_B.pth \
		--planner nonlearned \
		--env envB

.PHONY: lsp-planner-eval-C
lsp-planner-eval-C: DOCKER_ARGS ?= -it
lsp-planner-eval-C: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeABC_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_planner_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 4000 4100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/mazeC/mazeC.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2C/latest_net_G_B.pth \
		--planner nonlearned \
		--env envC

.PHONY: lsp-planner-results
lsp-planner-results: DOCKER_ARGS ?= -it
lsp-planner-results: arg-check-data
		@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_planner_results \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval

.PHONY: lsp-estimated-cost
lsp-estimated-cost: DOCKER_ARGS ?= -it
lsp-estimated-cost: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/delta_cost
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_estimated_cost \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 900 1000 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/delta_cost/ \
		--network_file /data/$(LSP_SELECT_BASENAME)/training_logs/maze/EnvA.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2B/latest_net_G_A.pth

.PHONY: lsp-rtm-lb-eval-A
lsp-rtm-lb-eval-A: DOCKER_ARGS ?= -it
lsp-rtm-lb-eval-A: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeABC_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_rtm_lb_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 2000 2100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--network_files mazeA/mazeA.pt mazeB/mazeB.pt mazeC/mazeC.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2A/latest_net_G_B.pth \
		--env envA

.PHONY: lsp-rtm-lb-eval-B
lsp-rtm-lb-eval-B: DOCKER_ARGS ?= -it
lsp-rtm-lb-eval-B: LSP_SELECT_UNITY_BASENAME = rail_sim_2022_07_greenfloor
lsp-rtm-lb-eval-B: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeABC_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_rtm_lb_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 3000 3100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--network_files mazeA/mazeA.pt mazeB/mazeB.pt mazeC/mazeC.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2B/latest_net_G_B.pth \
		--env envB

.PHONY: lsp-rtm-lb-eval-C
lsp-rtm-lb-eval-C: DOCKER_ARGS ?= -it
lsp-rtm-lb-eval-C: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/mazeABC_eval
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_rtm_lb_eval \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 4000 4100 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/mazeABC_eval \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--network_files mazeA/mazeA.pt mazeB/mazeB.pt mazeC/mazeC.pt \
		--generator_network_file /data/$(LSP_SELECT_BASENAME)/mazeA2C/latest_net_G_B.pth \
		--env envC

chosen_planner ?= nonlearned
sim_costs_dir ?= simulated_lb_costs
envA_name = mazeA
envB_name = office
envC_name = officewall

sim-costs-seeds-envA = \
$(shell for ii in $$(seq 2000 2060); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(sim_costs_dir)/target_$(chosen_planner)_$(envA_name)_$${ii}.txt"; done)
sim-costs-seeds-envB = \
$(shell for ii in $$(seq 3000 3060); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(sim_costs_dir)/target_$(chosen_planner)_$(envB_name)_$${ii}.txt"; done)
sim-costs-seeds-envC = \
$(shell for ii in $$(seq 4001 4061); \
		do echo "$(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(sim_costs_dir)/target_$(chosen_planner)_$(envC_name)_$${ii}.txt"; done)

.PHONY: lsp-simulated-costs-A
lsp-simulated-costs-A: $(sim-costs-seeds-envA)
$(sim-costs-seeds-envA): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(sim-costs-seeds-envA):
	$(call xhost_activate)
	@echo "Running [$(chosen_planner) | $(envA_name) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(sim_costs_dir)
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_simulated_costs \
		$(LSP_SELECT_CORE_ARGS) \
		--seed $(seed) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/$(sim_costs_dir) \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner $(chosen_planner) \
		--env $(envA_name)

.PHONY: lsp-simulated-costs-B
lsp-simulated-costs-B: $(sim-costs-seeds-envB)
$(sim-costs-seeds-envB): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(sim-costs-seeds-envB):
	$(call xhost_activate)
	@echo "Running [$(chosen_planner) | $(envB_name) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(sim_costs_dir)
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_simulated_costs \
		$(LSP_SELECT_CORE_ARGS) \
		--seed $(seed) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/$(sim_costs_dir) \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner $(chosen_planner) \
		--env $(envB_name)

.PHONY: lsp-simulated-costs-C
lsp-simulated-costs-C: LSP_SELECT_UNITY_BASENAME = rail_sim_2022_07_wallswap
lsp-simulated-costs-C: $(sim-costs-seeds-envC)
$(sim-costs-seeds-envC): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(sim-costs-seeds-envC):
	$(call xhost_activate)
	@echo "Running [$(chosen_planner) | $(envC_name) | seed: $(seed)]"
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/$(sim_costs_dir)
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_simulated_costs \
		$(LSP_SELECT_CORE_ARGS) \
		--seed $(seed) \
		--save_dir /data/$(LSP_SELECT_BASENAME)/$(sim_costs_dir) \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner $(chosen_planner) \
		--env $(envC_name)

.PHONY: lsp-planner-results-offline
lsp-planner-results-offline: DOCKER_ARGS ?= -it
lsp-planner-results-offline: xhost-activate arg-check-data
		@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_planner_results_offline \
		--save_dir /data/$(LSP_SELECT_BASENAME)/simulated_lb_costs \
		--experiment_name maze

.PHONY: lsp-simulate-planning
lsp-simulate-planning: DOCKER_ARGS ?= -it
lsp-simulate-planning: xhost-activate arg-check-data arg-check-unity
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_SELECT_BASENAME)/simulate_planning
	@$(DOCKER_PYTHON) -m lsp_select.scripts.lsp_simulate_planning \
		$(LSP_SELECT_CORE_ARGS) \
		--seed 3000 \
		--save_dir /data/$(LSP_SELECT_BASENAME)/simulate_planning \
		--network_path /data/$(LSP_SELECT_BASENAME)/training_logs \
		--chosen_planner lspoffice \
		--env office \
		--do_plot
