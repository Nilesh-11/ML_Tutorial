'''

The convolution operation is applied in a 2D manner, sliding a 2D filter 
(also known as a kernel) over the input image. Each element of the filter 
interacts with the corresponding elements in the input image, producing a 
weighted sum. This process is repeated across the entire image, creating a 
new output feature map.

torch.nn.Conv2d(in_channels,
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=0, 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='zeros', 
                device=None, dtype=None)

in_channels:    The input to the Conv2d layer is a 4D tensor representing a batch of images. 
                
    The dimensions are [batch_size, channels, height, width].
    
    batch_size:         The number of images in the batch.
    channels:           The number of channels in the input image. For grayscale images, this is 1. For color images (RGB), this is 3.
    height and width:   The spatial dimensions of the input image.

'''