.PHONY: build run

build:
	podman build -t piorosen/cprune .

run: build
	podman run -p 5911:22 --rm -v $(pwd):/work piorosen/cprune	
