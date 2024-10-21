#%%
import os
from dotenv import load_dotenv
import torchvision as tv
load_dotenv('/work/.env')
#%%
print(os.getenv('TORCH_HOME'))

#%%
tv.models.densenet121(True, True)
tv.models.densenet161(True, True)
tv.models.densenet201(True, True)
tv.models.alexnet(True, True)
tv.models.googlenet(True, True)
tv.models.resnet18(True, True)
tv.models.resnet34(True, True)
tv.models.resnet50(True, True)
tv.models.resnet101(True, True)
tv.models.resnet152(True, True)
tv.models.inception_v3(True, True)
tv.models.vgg11(True, True)
tv.models.vgg11_bn(True, True)
tv.models.vgg13(True, True)
tv.models.vgg13_bn(True, True)
tv.models.vgg16(True, True)
tv.models.vgg16_bn(True, True)
tv.models.vgg19(True, True)
tv.models.vgg19_bn(True, True)
tv.models.mobilenet_v3_large(True, True)
tv.models.mobilenet_v3_small(True, True)
tv.models.resnext50_32x4d(True, True)
tv.models.resnext101_32x8d(True, True)
tv.models.shufflenet_v2_x0_5(True, True)
tv.models.shufflenet_v2_x1_0
tv.models.shufflenet_v2_x1_5
tv.models.shufflenet_v2_x2_0
tv.models.squeezenet1_0
tv.models.squeezenet1_1
tv.models.



# %%
tv.__version__
# %%
