# Makefile for FlowNet package installation

# Define the base directory
BASE_DIR := ./respiration/extractor/flownet

# List of packages to install
PACKAGES := correlation_package resample2d_package channelnorm_package

# Default target
all: $(PACKAGES)

# Rule for each package
$(PACKAGES):
	@echo "Installing $@..."
	cd $(BASE_DIR)/$@ && \
	rm -rf *_cuda.egg-info build dist __pycache__ && \
	python3 setup.py install --user

# Clean target
clean:
	@echo "Cleaning up..."
	@for pkg in $(PACKAGES); do \
		rm -rf $(BASE_DIR)/$$pkg/*_cuda.egg-info $(BASE_DIR)/$$pkg/build $(BASE_DIR)/$$pkg/dist $(BASE_DIR)/$$pkg/__pycache__; \
	done

.PHONY: all $(PACKAGES) clean
