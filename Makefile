.PHONY: build publish test perf perf-flamegraph perf-profile bench-two-qubit-gates lint check suite prepare-windows-scratch

DIST_DIR := dist
REPOSITORY ?= testpypi
PYTHON ?= 3.11

ifeq ($(OS),Windows_NT)
ifneq (,$(filter //%,$(CURDIR)))
WINDOWS_SCRATCH ?= C:/codex-scratch/dqsim_bench
else
WINDOWS_SCRATCH ?= $(CURDIR)/.codex-win
endif
WINDOWS_SCRATCH := $(subst \,/,$(WINDOWS_SCRATCH))
export UV_CACHE_DIR ?= $(WINDOWS_SCRATCH)/uv-cache
export UV_PYTHON_INSTALL_DIR ?= $(WINDOWS_SCRATCH)/python
export UV_PROJECT_ENVIRONMENT ?= $(WINDOWS_SCRATCH)/venv
export CARGO_HOME ?= $(WINDOWS_SCRATCH)/cargo-home
export CARGO_TARGET_DIR ?= $(WINDOWS_SCRATCH)/target
export TMP := $(WINDOWS_SCRATCH)/tmp
export TEMP := $(WINDOWS_SCRATCH)/tmp
endif

ifeq ($(OS),Windows_NT)
prepare-windows-scratch:
	powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(WINDOWS_SCRATCH)/tmp' | Out-Null"
else
prepare-windows-scratch:
endif

ifeq ($(REPOSITORY),testpypi)
PUBLISH_URL := https://test.pypi.org/legacy/
else ifeq ($(REPOSITORY),pypi)
PUBLISH_URL := https://upload.pypi.org/legacy/
else
$(error Unsupported REPOSITORY '$(REPOSITORY)'; use REPOSITORY=testpypi or REPOSITORY=pypi)
endif

test: prepare-windows-scratch
	uv run --python $(PYTHON) --extra test --with maturin maturin develop --skip-install
	uv run --python $(PYTHON) --extra test python -m pytest tests/ -v -s

lint:
	cargo clippy --all-targets

check:
	cargo check

suite:
	$(MAKE) test && $(MAKE) lint && $(MAKE) check

perf: prepare-windows-scratch
	uv run --python $(PYTHON) --extra test --with maturin maturin develop --skip-install --release
	uv run --python $(PYTHON) --extra test python -m pytest benchmarking/benchmarking_suite.py -v -s

perf-flamegraph: prepare-windows-scratch
	rm -rf target/flamegraphs
	CARGO_PROFILE_RELEASE_DEBUG=true uv run --python $(PYTHON) --extra test --with maturin maturin develop --skip-install --release
	DQSIM_FLAMEGRAPH_DIR=target/flamegraphs uv run --python $(PYTHON) --extra test python -m pytest benchmarking/benchmarking_suite.py -v -s

perf-profile: perf-flamegraph

bench-two-qubit-gates:
	cargo bench --bench two_qubit_gates

build:
	rm -rf $(DIST_DIR)
	uvx --from build pyproject-build --outdir $(DIST_DIR)
	uvx twine check $(DIST_DIR)/*

publish: build
	uvx twine upload --repository-url $(PUBLISH_URL) $(DIST_DIR)/*
