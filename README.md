![banner cnns ppgcc ufsc](http://www.lapix.ufsc.br/wp-content/uploads/2019/06/VC-lapix.png)

# Weed-Mapping
Weed Mapping in Aerial Images through Identification and Segmentation of Crop Rows and Weeds using Convolutional Neural Networks


![alt-text-10](http://www.lapix.ufsc.br/wp-content/uploads/2019/06/weed-mapping.jpg)

## Description
This repository serves as a Weed Mapping Semantic Segmentation Suite. The goal is to easily be able to implement, train, and test new Semantic Segmentation models! 

It is based upon the, meanwhile deprecated, code repo at:  https://github.com/GeorgeSeif/Semantic-Segmentation-Suite. We, however, did not duplicate the whole repo and data here. Only the code that was necessary for our Weed Mapping apllication is here. Where modifications and extensions were needed, we did them. 

We also added a Jupyter Notebook with the whole high-level code necessary for training and predicting crop rows and weed areas. The datasets we employed in our experiments are here:

 - http://www.lapix.ufsc.br/weed-mapping-sugar-cane (Large Sugar Cane Field – Northern Brazil - contains weeds)
 - http://www.lapix.ufsc.br/crop-rows-sugar-cane (Large Sugar Cane Field – Northern Brazil - contains only well-behaved crops)

This code repo is complete with the following:

 - Jupyter Notebook with the whole high-level code necessary for training and predicting crop rows and weed areas
 - Training and testing modes
 - Data augmentation
 - Several state-of-the-art models. Easily **plug and play** with different models
 - Able to use **any other** dataset besides our own
 - Evaluation including precision, recall, f1 score, average accuracy, per-class accuracy, and mean IoU
 - Plotting of loss function and accuracy over epochs

**Any suggestions to improve this repository, including any new segmentation models you would like to see are welcome!**

## Frontends

The following feature extraction models are currently made available:

- [MobileNetV2](https://arxiv.org/abs/1801.04381), [ResNet50/101/152](https://arxiv.org/abs/1512.03385), and [InceptionV4](https://arxiv.org/abs/1602.07261)

## Models

The following segmentation models are currently made available:

- [Encoder-Decoder based on SegNet](https://arxiv.org/abs/1511.00561). This network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions.

- [Encoder-Decoder with skip connections based on SegNet](https://arxiv.org/abs/1511.00561). This network uses a VGG-style encoder-decoder, where the upsampling in the decoder is done using transposed convolutions. In addition, it employs additive skip connections from the encoder to the decoder. 

- [Mobile UNet for Semantic Segmentation](https://arxiv.org/abs/1704.04861). Combining the ideas of MobileNets Depthwise Separable Convolutions with UNet to build a high speed, low parameter Semantic Segmentation model.

- [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105). In this paper, the capability of global context information by different-region based context aggregation is applied through a pyramid pooling module together with the proposed pyramid scene parsing network (PSPNet). **Note that the original PSPNet uses a ResNet with dilated convolutions, but the one is this respository has only a regular ResNet.**

- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326). Uses a downsampling-upsampling style encoder-decoder network. Each stage i.e between the pooling layers uses dense blocks. In addition, it concatenated skip connections from the encoder to the decoder. In the code, this is the FC-DenseNet model.

- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587). This is the DeepLabV3 network. Uses Atrous Spatial Pyramid Pooling to capture multi-scale context by using multiple atrous rates. This creates a large receptive field. 

- [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612). A multi-path refinement network that explicitly exploits all the information available along the down-sampling process to enable high-resolution prediction using long-range residual connections. In this way, the deeper layers that capture high-level semantic features can be directly refined using fine-grained features from earlier convolutions.

- [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323). Combines multi-scale context with pixel-level accuracy by using two processing streams within the network. The residual stream carries information at the full image resolution, enabling precise adherence to segment boundaries. The pooling stream undergoes a sequence of pooling operations
to obtain robust features for recognition. The two streams are coupled at the full image resolution using residuals. In the code, this is the FRRN model.

- [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719). Proposes a Global Convolutional Network to address both the classification and localization issues for the semantic segmentation. Uses large separable kernals to expand the receptive field, plus a boundary refinement block to further improve localization performance near boundaries. 

- [AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions](http://ais.informatik.uni-freiburg.de/publications/papers/valada17icra.pdf) Modifies the ResNet50 architecture by performing the lower resolution processing using a multi-scale strategy with atrous convolutions. This is a slightly modified version using bilinear upscaling instead of transposed convolutions as I found it gave better results.

- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545). Proposes a compressed-PSPNet-based image cascade network (ICNet) that incorporates multi-resolution branches under proper label guidance to address this challenge. Most of the processing is done at low resolution for high speed and the multi-scale auxillary loss helps get an accurate model. **Note that for this model, I have implemented the network but have not integrated its training yet**

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611). This is the DeepLabV3+ network which adds a Decoder module on top of the regular DeepLabV3 model.

- [DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html). Combines many different scales using dilated convolution but with dense connections

- [Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/html/Bilinski_Dense_Decoder_Shortcut_CVPR_2018_paper.html). Dense Decoder Shorcut Connections using dense connectivity in the decoder stage of the segmentation model. **Note: this network takes a bit of extra time to load due to the construction of the ResNeXt blocks** 

- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897). BiSeNet use a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features while having a parallel Context Path with a fast downsampling strategy to obtain sufficient receptive field. 

- Or make your own and plug and play!


## Files and Directories

- **weedMaping.ipynb**: Jupyter Notebook with the whole high-level code necessary for training and predicting crop rows and weed areas

- **train.py:** Training on the dataset of your choice. Default is CamVid

- **test.py:** Testing on the dataset of your choice. Default is CamVid

- **predict.py:** Use your newly trained model to run a prediction on a single image

- **helper.py:** Quick helper functions for data preparation and visualization

- **utils.py:** Utilities for printing, debugging, testing, and evaluation

- **models:** Folder containing all model files. Use this to build your models, or use a pre-built one

- **CamVid:** The CamVid datatset for Semantic Segmentation as a test bed. This is the 32 class version

- **checkpoints:** Checkpoint files for each epoch during training

- **Test:** Test results including images, per-class accuracies, precision, recall, and f1 score


## Installation
This project has the following dependencies:

- Numpy `sudo pip install numpy`

- OpenCV Python `sudo apt-get install python-opencv`

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

## Usage
The only thing you have to do to get started is set up the folders in the following structure:

    ├── "dataset_name"                   
    |   ├── train
    |   ├── train_labels
    |   ├── val
    |   ├── val_labels
    |   ├── test
    |   ├── test_labels

Put a text file under the dataset directory called "class_dict.csv" which contains the list of classes along with the R, G, B colour labels to visualize the segmentation results. This kind of dictionairy is usually supplied with the dataset. Here is an example for the **Weed Mapping dataset**:

```
name,r,g,b
SugarCane,0,255,0
Soil,255,0,0
Invasive,255,255,0
```

**Note:** If you are using any of the networks that rely on a pre-trained ResNet, then you will need to download the pre-trained weights using the provided script. These are currently: PSPNet, RefineNet, DeepLabV3, DeepLabV3+, GCN.

Then you can simply run `train.py`! Check out the optional command line arguments:

```
usage: train.py [-h] [--num_epochs NUM_EPOCHS]
                [--checkpoint_step CHECKPOINT_STEP]
                [--validation_step VALIDATION_STEP] [--image IMAGE]
                [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                [--batch_size BATCH_SIZE] [--num_val_images NUM_VAL_IMAGES]
                [--h_flip H_FLIP] [--v_flip V_FLIP] [--brightness BRIGHTNESS]
                [--rotation ROTATION] [--model MODEL] [--frontend FRONTEND]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --checkpoint_step CHECKPOINT_STEP
                        How often to save checkpoints (epochs)
  --validation_step VALIDATION_STEP
                        How often to perform validation (epochs)
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation. Specifies the max bightness change
                        as a factor between 0.0 and 1.0. For example, 0.1
                        represents a max brightness change of 10% (+-).
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle in
                        degrees.
  --model MODEL         The model you are using. See model_builder.py for
                        supported models
  --frontend FRONTEND   The frontend you are using. See frontend_builder.py
                        for supported models

```
  
## Citing this Git


```tex
@misc{WeedMappingCode2019,
  author = {Monteiro, A.A.O. and von Wangenheim, A.},
  title = {Weed Mapping in Aerial Images through Identification and Segmentation of Crop Rows and Weeds},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/awangenh/Weed-Mapping}}
}
```

![banner Creative Commons INCoD UFSC](http://www.lapix.ufsc.br/wp-content/uploads/2019/05/cc.png)

