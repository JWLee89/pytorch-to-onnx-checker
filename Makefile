help:
	@echo 'Usage: `make <target>` (see target list below)'
	@echo
	@sed -e '/#\{2\}-/!d; s/\\$$//; s/:[^#\t]*/:/; s/#\{2\}-*//' $(MAKEFILE_LIST)

# Load the .env file configs so that our env file is loaded properly
.PHONY: load-env
load-env:
		@$(eval load_env=set -a;test -f ./.env && . ./.env;set +a)

# ------------ Docker build commands ------------
# -----------------------------------------------
# FYI: feel free to update the :latest tag with custom tags
# when customizing your projects

.PHONY: build-base-docker
build-base-docker: load-env ##- Build a docker image up to the "base" stage.
	@$(load_env); docker build \
		--target base \
		--tag ${USER}/pt2onnx_checker-base:latest \
		.

# Build a docker image up to the "check" stage
# -----------------------------
.PHONY: build-check-docker
build-check-docker: load-env ##- Build a docker image for check.
	@$(load_env); docker build \
		--target check \
		--tag ${USER}/pt2onnx_checker-check:latest \
		.

.PHONY: build-dev-docker
build-dev-docker: load-env ##- ##- Build a docker image up to the "dev" stage
	@$(eval uid=$(shell id -u))
	@$(load_env); docker build \
		--build-arg USER_ID=${uid} \
		--build-arg USER_NAME=${USER} \
		--build-arg USER_HOME=${HOME} \
		--build-arg GROUP_ID=9001 \
		--build-arg GROUP_NAME=user \
		--target dev \
		--tag ${USER}/pt2onnx_checker-dev:latest \
		.

# TODO: For each project, create appropriate commands

# ------------- Docker run commands -------------
# -----------------------------------------------

.PHONY: run-docker
run-docker: load-env ##- Run a docker container for development.
# Being highly related and frequently tested in their compatibility, insight-inference-postprocess is also mounted.
	@$(load_env); docker run -itd \
		-w ${HOME}/workspace/pytorch-to-onnx-checker \
		-v ${PWD}/:${HOME}/workspace/pytorch-to-onnx-checker \
		--shm-size=64g \
		--name ${USER}_pytorch-to-onnx-checker_dev \
		${USER}/pt2onnx_checker-dev:latest \
		zsh

.PHONY: run-docker-local
run-docker-local: load-env ##- Run a docker container for development on local device (e.g. Mac)
	@$(load_env); docker run -itd \
		-w ${HOME}/workspace/pytorch-to-onnx-checker \
		-v ${PWD}/:${HOME}/workspace/pytorch-to-onnx-checker \
		--shm-size=64g \
		--name ${USER}_pytorch-to-onnx-checker_dev \
		${USER}/pt2onnx_checker-dev:latest \
		zsh

.PHONY: attach-docker
attach-docker: load-env ##- Attach to docker container for development
	@docker exec -it ${USER}_pytorch-to-onnx-checker_dev zsh

.PHONY: run-tests
run-tests: load-env	##- Run the tests
	@$(load_env); pytest --cov pt2onnx_checker --cov-report=xml

.PHONY: run-fast-tests
run-fast-tests: load-env	##- Run the "now slow" tests
	@$(load_env); pytest -k "not slow"
