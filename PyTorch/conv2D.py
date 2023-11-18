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

in_channels:    channels(channels is defined below)
 
    The dimensions of input [batch_size, channels, height, width].
    batch_size:         The number of images in the batch.
    channels:           The number of channels in the input image. For grayscale images, this is 1. For color images (RGB), this is 3.
    height and width:   The spatial dimensions of the input image.

'''

'''
torch.nn.BatchNorm2d(num_features, 
                     eps=1e-05, 
                     momentum=0.1, 
                     affine=True, 
                     track_running_stats=True, 
                     device=None, 
                     dtype=None)
'''

'''
LAUNCH TENSORBOARD:

    # Initialize the model and TensorBoard writer
    model = BasicBlock(in_planes=3, planes=64)
    dummy_input = torch.randn(1, 3, 224, 224)  # Create a dummy input for visualization
    with SummaryWriter(comment='classname') as writer:
        writer.add_graph(model, dummy_input)

    # Run your script and then launch TensorBoard using the command:
    # tensorboard --logdir=runs
'''