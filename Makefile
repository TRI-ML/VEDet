WORK_DIR=${PWD}
PROJECT=vedet
DOCKER_IMAGE=${PROJECT}:latest
DOCKER_FILE=docker/Dockerfile-mmlab-cu111
DATA_ROOT?=/mnt/fsx-2/datasets
CKPTS_ROOT?=/mnt/fsx-2/ckpts
SAVE_ROOT?=/mnt/fsx-2/experiments

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ~/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws \
	-v ${WORK_DIR}:/workspace/${PROJECT} \
	-v ${DATA_ROOT}:/workspace/${PROJECT}/data \
	-v ${CKPTS_ROOT}:/workspace/${PROJECT}/ckpts \
	-v ${SAVE_ROOT}:/workspace/${PROJECT}/work_dirs \
	--shm-size=8G \
	--ipc=host \
	--network=host \
	--pid=host \
	--privileged

DOCKER_BUILD_ARGS = \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \

docker-build:
	nvidia-docker image build -f $(DOCKER_FILE) -t $(DOCKER_IMAGE) \
	$(DOCKER_BUILD_ARGS) .

docker-dev:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf
