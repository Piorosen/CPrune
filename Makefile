.PHONY: build run

build:
	docker build -t piorosen/cprune .

run: 
	docker run -p 5911:22 --rm -v $(shell pwd):/work piorosen/cprune	
# bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

recover:
	chown -R 2004:2000 *

in_con: 
	chown -R 0:0 *

download_cifar:

download_imagenet:
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
