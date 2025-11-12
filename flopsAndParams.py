# from torchvision.models import resnet34
from CDCNet import CDC_Base as CDC
import torch
from thop import profile
if __name__ == "__main__":

    model = CDC(8)  #6, 7, 8
    input = torch.randn(1, 3, 224, 224)
    Flops, params = profile(model, inputs=(input,))
    print('Flops: % .4fG'%(Flops / 1000000000))
    print('params: % .4fM'% (params / 1000000))