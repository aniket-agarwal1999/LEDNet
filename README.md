# LEDNet

#### A Lightweight Encoder-Decoder Network for real-time semantic segmentation

This is the unofficial implementation of the paper [LEDNet](https://arxiv.org/pdf/1905.02423.pdf).

This repo contains the whole model architecture for the LEDNet model. You can also view the soon to be released official implementation at this [repo](https://github.com/xiaoyufenfei/LEDNet). Apart from this, I have tried to replicate the model to the very best given in the paper and have even used some inspiration for the missing information from the [ENet model](https://arxiv.org/pdf/1606.02147.pdf). Hope you will find this model useful :smile: 

### Current Status of the project
Currently this repo contains the directory model containing the whole model architecture excluding the model architecture. So to use as a model in your segmentation task you can just place the directory `model` in your working directory. After that

```python
from model import return_model
....
# So the below line will initialize the LEDNet model 128*128 images
seg_model = return_model(input_nc = 3, output_nc = 22, netG = 'lednet_128')
# Also the input_channels and output_channels can be handled accordingly
```
Also the **model architecture has already been tested**. Soon I will **update the training and testing procedure for the VOC segmentation task and also for the Cityscapes dataset**.

### Specifications of the architecture
Although most of the things have been taken up directly from what was specified in the original paper, but due to some changes it is best to specify them here:
1. As specified in the original Enet paper, I have used `PReLu` activation in the encoder part, but used `ReLu` in the decoder part. However have bias terms in all of the network.
2. In every downsampling block after the concatenation of the two parallel operations there is application of activation.
3. Also `BatchNorm2d` has been in every `SSnbt` module, as the results were not that great in its absence.
4. There is application of `Dropout2d` in every `SSnbt` module after concatenation of its left and right branch.
5. Most importantly, for the upsampling in the end and also in the `APN` module, I have used `Bilinear Interpolation`. I also tried using `ConvTraspose2d` initially but it lead very poor results and also checkered effects in the final results.
