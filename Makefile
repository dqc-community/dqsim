.PHONY: build publish test perf perf-flamegraph perf-profile bench-two-qubit-gates lint check suite

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

perf-flamegraph:
	rm -rf target/flamegraphs
	CARGO_PROFILE_RELEASE_DEBUG=true uvx --python 3.11 maturin develop --skip-install --release
	DQSIM_FLAMEGRAPH_DIR=target/flamegraphs uv run --python 3.11 --extra test pytest benchmarking/benchmarking_suite.py -v -s

perf-profile: perf-flamegraph

bench-two-qubit-gates:
	cargo bench --bench two_qubit_gates

build:
	rm -rf $(DIST_DIR)
	uvx --from build pyproject-build --outdir $(DIST_DIR)
	uvx twine check $(DIST_DIR)/*

publish: build
	uvx twine upload --repository-url $(PUBLISH_URL) $(DIST_DIR)/*
