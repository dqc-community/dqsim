.PHONY: build publish test perf perf-profile lint check suite

DIST_DIR := dist
REPOSITORY ?= testpypi

ifeq ($(REPOSITORY),testpypi)
PUBLISH_URL := https://test.pypi.org/legacy/
else ifeq ($(REPOSITORY),pypi)
PUBLISH_URL := https://upload.pypi.org/legacy/
else
$(error Unsupported REPOSITORY '$(REPOSITORY)'; use REPOSITORY=testpypi or REPOSITORY=pypi)
endif

test:
	uvx --python 3.11 maturin develop --skip-install
	uv run --python 3.11 --extra test pytest tests/ -v -s

lint:
	cargo clippy --all-targets

check:
	cargo check

suite:
	$(MAKE) test && $(MAKE) lint && $(MAKE) check

perf:
	uvx --python 3.11 maturin develop --skip-install --release
	uv run --python 3.11 --extra test pytest benchmarking/benchmarking_suite.py -v -s

perf-profile:
	uvx --python 3.11 maturin develop --skip-install --release
	DQSIM_PROFILE_SHOTS=1 uv run --python 3.11 --extra test pytest benchmarking/benchmarking_suite.py -v -s

build:
	rm -rf $(DIST_DIR)
	uvx --from build pyproject-build --outdir $(DIST_DIR)
	uvx twine check $(DIST_DIR)/*

publish: build
	uvx twine upload --repository-url $(PUBLISH_URL) $(DIST_DIR)/*
