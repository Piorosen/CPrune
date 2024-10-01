.PHONY: build run

build:
	docker build -t chacha/cprune:3.12 .

run: 
	sudo docker run --network host -d -v $(pwd):/work chacha/cprune:3.12
# bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

recover:
	chown -R 1003:1003 *

in_con: 
	chown -R 0:0 *

download_cifar:

download_imagenet:
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar && \
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar && \
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && \
	wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
