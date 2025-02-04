from torch import nn

def PDNs(out_channels:int=384, padding:bool=False) -> nn.Sequential:
    pad_mult = 1 if padding == True else 0
    return nn.Sequential(
        # Conv-1
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=3*pad_mult),
        nn.ReLU(inplace=True),
        # AvgPool-1
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1*pad_mult),
        # Conv-2
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=3*pad_mult),
        nn.ReLU(inplace=True),
        # AvgPool-2
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1*pad_mult),
        # Conv-3
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1*pad_mult),
        nn.ReLU(inplace=True),
        # Conv-4
        nn.Conv2d(in_channels=256, out_channels=out_channels, stride=1, kernel_size=4, padding=0)
    )

def PDNm(out_channels:int=384, padding:bool=False) -> nn.Sequential:
    pad_mult = 1 if padding == True else 0
    return nn.Sequential(
        # Conv-1
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=1, padding=3*pad_mult),
        nn.ReLU(inplace=True),
        # AvgPool-1
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1*pad_mult),
        # Conv-2
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=3*pad_mult),
        nn.ReLU(inplace=True),
        # AvgPool-2
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1*pad_mult),
        # Conv-3
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
        nn.ReLU(inplace=True),
        # Conv-4
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1*pad_mult),
        nn.ReLU(inplace=True),
        # Conv-5
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4, stride=1),
        nn.ReLU(inplace=True),
        # Conv-6
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
    )
    
def create_model(model_size: str, out_channels: int):
    """
    Args:
        model_size: 'PDNs' or 'PDNm'
        out_channels: number of output channels
    """
    if model_size == 'small':
        teacher = PDNs(out_channels=out_channels)
        student = PDNs(out_channels=2*out_channels)
        # teacher = PDNs(out_channels, padding=True)    #FIXME: teacher output dimension : (1, 384, 64, 64)
        # student = PDNs(2 * out_channels, padding=True)#FIXME: autoencoder output dimension : (1, 384, 56, 56)
    elif model_size == 'medium':
        teacher = PDNm(out_channels=out_channels)
        student = PDNm(out_channels=2*out_channels)
        # teacher = PDNm(out_channels, padding=True)    #FIXME: teacher output dimension : (1, 384, 64, 64)
        # student = PDNm(2 * out_channels, padding=True)#FIXME: autoencoder output dimension : (1, 384, 56, 56)
    else:
        raise ValueError('model_size should be either "small" or "medium"')
    
    return teacher, student