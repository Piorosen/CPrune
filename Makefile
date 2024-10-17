.PHONY: build run

build:
	sudo docker build -t chacha/cprune:3.8 .

run: 
	sudo docker run --gpus all -p 5911:5911 -d -v $(shell pwd):/work chacha/cprune:3.8

it:
	sudo docker run --gpus all --rm -it -v $(shell pwd):/work chacha/cprune:3.8 /bin/bash

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
