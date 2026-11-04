SHELL := /bin/bash

ENVS_DIR ?= envs
PYTHON_VERSION ?= 3.12
TOOLS := $(ENVS_DIR)/tools

NVCC ?= $(if $(CUDA_HOME),$(CUDA_HOME)/bin/nvcc,/usr/local/cuda/bin/nvcc)
CUDA_INC ?= $(if $(CUDA_HOME),$(CUDA_HOME)/include,/usr/local/cuda/include)

.PHONY: setup check code envs clean help
.DEFAULT_GOAL := help

setup: ## build the tools venv (pytest, ruff, numpy); run once, needs network
	@ENVS_DIR="$(ENVS_DIR)" PYTHON_VERSION="$(PYTHON_VERSION)" bash scripts/setup-envs.sh tools

check: ## run the unit tests and the linter (hermetic: no GPU, no network)
	@test -x "$(TOOLS)/bin/python" || { echo "run 'make setup' once first (then 'make check' is offline)" >&2; exit 1; }
	@"$(TOOLS)/bin/python" -m pytest tests/unit
	@"$(TOOLS)/bin/ruff" check kvguard tests
	@"$(TOOLS)/bin/ruff" format --check kvguard tests

code: ## build the optional GPU toolchain (interposer .so, validator PTX, nvtx shim); needs CUDA + CMake + Ninja, no GPU
	@cmake -S interposer -B build/interposer -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
	@cmake --build build/interposer
	@mkdir -p build/validators
	@"$(NVCC)" -arch=sm_80 -O3 -ptx kvguard/gpu/validators/validators.cu -o build/validators/validators.ptx
	@mkdir -p build/nvtx
	@gcc -shared -fPIC -O2 -I"$(CUDA_INC)" kvguard/gpu/nvtxmem/kvg_nvtx.c -o build/nvtx/libkvg_nvtx.so
	@echo "built build/interposer/libkvg_interposer.so, build/validators/validators.ptx, build/nvtx/libkvg_nvtx.so"

envs: ## build the vLLM 0.32.0 and SGLang 0.5.22 environments (network, ~10 GB each)
	@ENVS_DIR="$(ENVS_DIR)" PYTHON_VERSION="$(PYTHON_VERSION)" bash scripts/setup-envs.sh vllm-0.32.0 sglang-0.5.22

clean: ## remove build artifacts
	@rm -rf build

help: ## list available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' Makefile \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-8s\033[0m %s\n", $$1, $$2}'
