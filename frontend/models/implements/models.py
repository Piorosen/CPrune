#%%
import os
import torchvision as tv

def get_model_zoo():
    checkpoint_dir = os.path.join(os.getenv("TORCH_HOME"), 'hub', 'checkpoints')
    model_dict = {
        "alexnet": [tv.models.alexnet, os.path.join(checkpoint_dir, 'alexnet-owt-4df8aa71.pth')],
        "densenet121": [tv.models.densenet121, os.path.join(checkpoint_dir, 'densenet121-a639ec97.pth')],
        "densenet161": [tv.models.densenet161, os.path.join(checkpoint_dir, 'densenet161-8d451a50.pth')],
        "densenet201": [tv.models.densenet201, os.path.join(checkpoint_dir, 'densenet201-c1103571.pth')],
        "googlenet": [tv.models.googlenet, os.path.join(checkpoint_dir, 'googlenet-1378be20.pth')],
        "inception_v3": [tv.models.inception_v3, os.path.join(checkpoint_dir, 'inception_v3_google-1a9a5a14.pth')],
        "mobilenet_v2": [tv.models.mobilenet_v2, os.path.join(checkpoint_dir, 'mobilenet_v2-b0353104.pth')],
        "resnet18": [tv.models.resnet18, os.path.join(checkpoint_dir, 'resnet18-5c106cde.pth')],
        "resnet34": [tv.models.resnet34, os.path.join(checkpoint_dir, 'resnet34-333f7ec4.pth')],
        "resnet50": [tv.models.resnet50, os.path.join(checkpoint_dir, 'resnet50-19c8e357.pth')],
        "resnet101": [tv.models.resnet101, os.path.join(checkpoint_dir, 'resnet101-5d3b4d8f.pth')],
        "resnet152": [tv.models.resnet152, os.path.join(checkpoint_dir, 'resnet152-b121ed2d.pth')],
        "resnext50_32x4d": [tv.models.resnext50_32x4d, os.path.join(checkpoint_dir, 'resnext50_32x4d-7cdf4587.pth')],
        "resnext101_32x8d": [tv.models.resnext101_32x8d, os.path.join(checkpoint_dir, 'resnext101_32x8d-8ba56ff5.pth')],
        "shufflenet_v2_x0_5": [tv.models.shufflenet_v2_x0_5, os.path.join(checkpoint_dir, 'shufflenetv2_x0.5-F707e7162e.pth')],
        "shufflenet_v2_x1_0": [tv.models.shufflenet_v2_x1_0, os.path.join(checkpoint_dir, 'shufflenetv2_x1-5666bf0f80.pth')],
        "squeezenet1_0": [tv.models.squeezenet1_0, os.path.join(checkpoint_dir, 'squeezenet1_0-a815701f.pth')],
        "squeezenet1_1": [tv.models.squeezenet1_1, os.path.join(checkpoint_dir, 'squeezenet1_1-f364aa15.pth')],
        "vgg11": [tv.models.vgg11, os.path.join(checkpoint_dir, 'vgg11-bbd30ac9.pth')],
        "vgg11_bn": [tv.models.vgg11_bn, os.path.join(checkpoint_dir, 'vgg11_bn-6002323d.pth')],
        "vgg13": [tv.models.vgg13, os.path.join(checkpoint_dir, 'vgg13-c768596a.pth')],
        "vgg13_bn": [tv.models.vgg13_bn, os.path.join(checkpoint_dir, 'vgg13_bn-abd245e5.pth')],
        "vgg16": [tv.models.vgg16, os.path.join(checkpoint_dir, 'vgg16-397923af.pth')],
        "vgg16_bn": [tv.models.vgg16_bn, os.path.join(checkpoint_dir, 'vgg16_bn-6c64b313.pth')],
        "vgg19": [tv.models.vgg19, os.path.join(checkpoint_dir, 'vgg19-bcbb9e9d.pth')],
        "vgg19_bn": [tv.models.vgg19_bn, os.path.join(checkpoint_dir, 'vgg19_bn-c79401a0.pth')],
    }
    return model_dict

