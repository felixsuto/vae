#!/bin/bash
docker build --no-cache \
--build-arg USRNM=$(whoami) \
--build-arg USRUID=$(id -u) \
--build-arg USRGID=$(id -g) \
-t vae:latest .
