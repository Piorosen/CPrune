from enum import Enum, auto

import models.cifar10 as cifar10
import models.imagenet as imgnet
import models.mnist as mnist
import torchvision.models as tmodels

class storage(Enum):
    CIFAR10_RESNET18 = auto()
    CIFAR10_RESNET34 = auto()
    CIFAR10_RESNET50 = auto()
    CIFAR10_RESNET101 = auto()
    CIFAR10_RESNET152 = auto()
    
    CIFAR10_VGG11 = auto()
    CIFAR10_VGG13 = auto()
    CIFAR10_VGG16 = auto()
    CIFAR10_VGG19 = auto()
    
    IMAGENET_MOBILENET_V1 = auto()
    IMAGENET_MOBILENET_V2 = auto()
    
    IMAGENET_VGG11 = auto()
    IMAGENET_VGG13 = auto()
    IMAGENET_VGG16 = auto()
    IMAGENET_VGG19 = auto()
    
    MNIST_LENET = auto()
    OTHERS_FROM_TORCH = auto()
    

def load(data: storage):
    # CIFAR-10 models
    if data == storage.CIFAR10_RESNET18:
        return cifar10.ResNet18()
    elif data == storage.CIFAR10_RESNET34:
        return cifar10.ResNet34()
    elif data == storage.CIFAR10_RESNET50:
        return cifar10.ResNet50()
    elif data == storage.CIFAR10_RESNET101:
        return cifar10.ResNet101()
    elif data == storage.CIFAR10_RESNET152:
        return cifar10.ResNet152()
    elif data in {storage.CIFAR10_VGG11, storage.CIFAR10_VGG13, storage.CIFAR10_VGG16, storage.CIFAR10_VGG19}:
        return getattr(cifar10, data.name.split('_')[1])()  # Dynamically loads VGG models

    # ImageNet models
    elif data == storage.IMAGENET_MOBILENET_V1:
        return imgnet.MobileNetV1()
    elif data == storage.IMAGENET_MOBILENET_V2:
        return imgnet.MobileNetV2()
    elif data in {storage.IMAGENET_VGG11, storage.IMAGENET_VGG13, storage.IMAGENET_VGG16, storage.IMAGENET_VGG19}:
        return getattr(imgnet, data.name.split('_')[1])()  # Dynamically loads VGG models

    # MNIST models
    elif data == storage.MNIST_LENET:
        return mnist.LeNet()
    elif data == storage.OTHERS_FROM_TORCH:
        return tmodels
    else:
        raise ValueError(f"Model {data} not found.")
