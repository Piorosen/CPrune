.PHONY: build run

build:
	docker build -t piorosen/cprune .

run: 
	docker run -p 5911:22 --rm -v $(shell pwd):/work piorosen/cprune	
# bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
