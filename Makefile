CURRENT_DIR := $(shell pwd)
APP_NAME := $(shell basename $(CURRENT_DIR))

REGISTRY := javohirgo
IMAGE_NAME := smile-mini-pc-app
TAG := latest

DOCKERFILE := Dockerfile

build-image:
	docker build --rm -f $(DOCKERFILE) -t $(REGISTRY)/$(IMAGE_NAME):$(TAG) .

push-image:
	docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)

clear-image:
	docker rmi $(REGISTRY)/$(IMAGE_NAME):$(TAG)
