# from torchvision.models import resnet34
from CDCNet import CDC_Base as CDC
import torch
from thop import profile
if __name__ == "__main__":

    model = CDC(8)
    input = torch.randn(1, 3, 224, 224)
    Flops, params = profile(model, inputs=(input,)) # macs
    print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值