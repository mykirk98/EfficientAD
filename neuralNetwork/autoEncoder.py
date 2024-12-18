from torch import nn

def AutoEncoder(out_channels:int=384) -> nn.Sequential:
    return nn.Sequential(
        # ENCODER
        # EncConv-1
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # EncConv-2
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # EncConv-3
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # EncConv-4
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # EncConv-5
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # EncConv-6
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding=0),
        
        # DECODER
        # Bilinear-1
        nn.Upsample(size=3, mode='bilinear'),
        # DecConv-1
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        # Dropout-1
        nn.Dropout(p=0.2),  #FIXME: what about inplace=True?
        # Bilinear-2
        nn.Upsample(size=8, mode='bilinear'),
        # DecConv-2
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        # Dropout-2
        nn.Dropout(p=0.2),
        # Bilinear-3
        nn.Upsample(size=15, mode='bilinear'),
        # DecConv-3
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        # Dropout-3
        nn.Dropout(p=0.2),
        # Bilinear-4
        nn.Upsample(size=32, mode='bilinear'),
        # DecConv-4
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        # Dropout-4
        nn.Dropout(p=0.2),
        # Bilinear-5
        nn.Upsample(size=63, mode='bilinear'),
        # DecConv-5
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        # Dropout-5
        nn.Dropout(p=0.2),
        # Bilinear-6
        nn.Upsample(size=127, mode='bilinear'),
        # DecConv-6
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        # Dropout-6
        nn.Dropout(p=0.2),
        # Biliear-7
        nn.Upsample(size=56, mode='bilinear'),
        # nn.Upsample(size=64, mode='bilinear'),    #FIXME: 논문에서는 128 -> 646로 업샘플링, 따라서 size=64여야함
        # DecConv-7
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        # DecConv-8
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
    )