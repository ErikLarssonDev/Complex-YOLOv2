## Work with Nvidia GPU inside a container
For installing the docker container gpu toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Restart docker engine after installing the toolkit with `systemctl`

Then have a test by

```shell
docker run --gpus all nvidia/cuda:11.2.2-base-ubuntu16.04 nvidia-smi
```

You can find more images with different version of cuda in [Nvidia's docker hub](https://hub.docker.com/r/nvidia/cuda/)

## Start a container

```shell
docker run -it --gpus all -v {dir_on_host}:{dir_in_container} {iamge_name}
```

then maybe test to install some dependencies.

## Dockerfile

After testing and everything is ok, you can write a dockerfile like

```dockerfile
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       zlib1g-dev \
                       vim \
                       python3 \
                       python3-pip

RUN pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html 

RUN mkdir /workspace

WORKDIR '/workspace'

ENTRYPOINT ["/bin/bash"]
```

More can find in the [doc of docker](https://docs.docker.com/build/guide/intro/)

The default name of a dockerfile is `Dockerfile`, after creating it, run cmd like below to build a image 

```shell
docker build -t yolov2-image-2 .
```

then create a container with the built image, like

```shell
docker run -it   --gpus 'all'   -v "${PWD}:/workspace"    -v "/media/storage:/workspace/dataset" -v "/home/student/minzod_mmdet3d:/workspace/minzod_mmdet3d" -v "/home/student/minizod:/workspace/minizod" --name "yolov2-container" --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  yolov2-image
```
or
```shell
docker run -it   --gpus 'all'   -v "${PWD}:/workspace"   -v "/media/dataSsd/KITTI:/workspace/KITTI"   -v "/home/student/minzod_mmdet3d:/workspace/minzod_mmdet3d" -v "/media/dataSsd/zod:/workspace/zod"   --name "yolov2-container-2" --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  yolov2-image-2
```

it might be important to add "X11Forwarding yes" to your ssh_config (found in etc/)
also make sure to run ´sudo xhost +´ on the host machine and that the $DISPLAY variable of the container matches the one of the host 

when container is build run 
```shell
apt-get update -y
apt-get install libglib2.0-0 -y # 8 49
apt-get install libsm6 libxrender1 libfontconfig1
pip install --no-cache-dir -r requirements.txt
```

Start container again: 
```shell
docker start -i yolov2-container
```

Remove all containers:

```shell
docker rm $(docker ps -a -q)
```

## Work with VSCode

There are two extensions will make it more convenience to work with docker container. They are `Dev Containers` and `Docker`. They will let you attach a vscode window on a container.
