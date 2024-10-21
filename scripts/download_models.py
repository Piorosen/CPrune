#%%
import os
from dotenv import load_dotenv
import torchvision as tv

import os
import argparse


def main():
    # Initialize argparse to read arguments from the terminal
    parser = argparse.ArgumentParser(description="Read and print TORCH_HOME environment variable.")
    parser.add_argument('--torch_home', type=str, help="Path to set as TORCH_HOME", required=False)

    # Parse the arguments
    args, unknown = parser.parse_known_args()

    # If --torch_home is provided, set it as the TORCH_HOME environment variable
    if args.torch_home:
        os.environ['TORCH_HOME'] = args.torch_home
        print(f"TORCH_HOME is set to: {os.environ['TORCH_HOME']}")
    else:
        # If --torch_home is not provided, read the current value of TORCH_HOME
        torch_home = os.getenv('TORCH_HOME', "TORCH_HOME is not set")
        print(f"Current TORCH_HOME: {torch_home}")

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
    tv.models.mobilenet_v2(True, True)
    tv.models.resnext50_32x4d(True, True)
    tv.models.resnext101_32x8d(True, True)
    tv.models.shufflenet_v2_x0_5(True, True)
    tv.models.shufflenet_v2_x1_0(True, True)
    # tv.models.shufflenet_v2_x1_5(True, True)
    # tv.models.shufflenet_v2_x2_0(True, True)
    tv.models.squeezenet1_0(True, True)
    tv.models.squeezenet1_1(True, True)
    
if __name__ == "__main__":
    main()

# %%
