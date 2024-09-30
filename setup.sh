#!/bin/bash

podman build -t piorosen/cprune .
podman run -p 5911:22 --rm -v $(pwd):/work piorosen/cprune

