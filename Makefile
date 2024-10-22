.PHONY: build run

build:
	docker build -t chacha/cprune:3.8-cuda .

run: 
	docker run --gpus all --shm-size=250G --network=host -d -v $(shell pwd):/work chacha/cprune:3.8-cuda

it:
	docker run --gpus all --rm --shm-size=250G -it -v $(shell pwd):/work chacha/cprune:3.8-cuda /bin/bash

# bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

recover:
	chown -R 1003:1003 *

in_con: 
	chown -R 0:0 *

download_cifar:

download_imagenet:
	wget -P ./dataset/imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar && \
	wget -P ./dataset/imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar && \
	wget -P ./dataset/imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && \
	wget -P ./dataset/imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar && \
	wget -P ./dataset/imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t3.tar.gz && \
	wget -P ./dataset/imagenet https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz	
