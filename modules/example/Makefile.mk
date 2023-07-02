
help::
	@echo "Example Targets (from the example module)"
	@echo "  example-core	Create a simple example image"
	@echo "  example-clean	Deletes the simple example image"
	@echo ""

# This target is to make an image by calling a script within 'example'
example-image = $(DATA_BASE_DIR)/example_module/demo_image.png
$(example-image):
	@mkdir -p $(DATA_BASE_DIR)/example_module
	@$(DOCKER_PYTHON) -m example.scripts.gen_random_mat \
		--image-name /data/example_module/demo_image.png

# A high-level target that calls other targets with a more convenient name
.PHONY: example-core
example-core: $(example-image)

example-clean:
	@echo "Cleaning products from 'example' repo."
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@-$(DOCKER_BASE) rm -r /data/example_module
	@echo "...done cleaning."

