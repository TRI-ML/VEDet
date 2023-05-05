## Use Docker Environment
We provide a self-contained dockerfile and recommend preparing the environment using docker. To build the image, run the following command in this directory:
```bash
make docker-build
```

We use [Weights & Biases](https://wandb.ai/site) to log the training. To optionally build your wandb credentials into the docker, run (if you don't build credentials into the docker, you can still manually log into wandb after entering the docker container):
```bash
make docker-build WANDB_API_KEY=<your WANDB_API_KEY> WANDB_ENTITY=<your WANDB_ENTITY>
```

After the image is built, run the following command with your paths on the host machine for data, checkpoints, and logging, to enter the dockerized environment:
```bash
# DATA_ROOT will be mounted as /workspace/vedet/data
# CKPTS_ROOT will be mounted as /workspace/vedet/ckpts
# SAVE_ROOT will be mounted as /workspace/vedet/work_dirs
make docker-dev DATA_ROOT=<host DATA_ROOT> CKPTS_ROOT=<host CKPTS_ROOT> SAVE_ROOT=<host SAVE_ROOT>
```

Inside the docker the folder structure will look like this, with data, checkpoints, logging paths mounted under `/workspace/vedet/`:
```
/workspace/
|-- mmlab
|   |-- mmdetection
|   |-- mmdetection3d
|   `-- mmsegmentation
`-- vedet
    |-- LICENSE.md
    |-- Makefile
    |-- README.md
    |-- ckpts
    |-- data
    |-- docker
    |-- docs
    |-- projects
    |-- requirements.txt
    |-- tools
```

## Use Pip/Conda Environment
The pytorch version we use in this project is `1.9.0` with CUDA `11.1`, CUDNN `8`. After install the right version in your environment, please install the following dependencies.

### Python tools
```bash
pip install \
    wandb==0.12.17 \
    einops==0.4.1 \
    pytorch3d==0.3.0 \
    pycocotools==2.0.4 \
    nuscenes-devkit==1.1.7 \
    timm==0.6.11
```

### OpenMMLab packages
```bash
export MMCV="1.4.0"
export MMDET="v2.25.0"
export MMSEG="v0.20.2"
export MMDET3D="v0.17.1"
export FORCE_CUDA="1"

# install mmcv
pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout ${MMDET}
pip install -r requirements/build.txt && pip install -e .

# install mmsegmentation
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation && git checkout ${MMSEG}
pip install -e .

# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout ${MMDET3D}
pip install -e .
```

### Data, checkpoints, logging paths
```bash
# enter the project top-level directory
cd vedet
ln -s $DATA_ROOT data/
ln -s $CKPTS_ROOT ckpts/
ln -s $SAVE_ROOT work_dirs/
```
