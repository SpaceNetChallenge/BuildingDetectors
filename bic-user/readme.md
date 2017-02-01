## Topcoder: [bic-user's](https://www.topcoder.com/members/bic-user/) implementation

#### Competition: [SpaceNet Challenge 1 - Rio de Janeiro Building Footprint Extraction](http://crowdsourcing.topcoder.com/spacenet)

#### Rank in competition: 5

#### Official [Building Footprint Metric](https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb#.q0v9inh3i) Score: 0.168605

#### Dataset: [SpaceNet AWS S3](https://aws.amazon.com/public-datasets/spacenet/)

#### Approach:
This implementation uses a fully convolutional network (FCN), based on this [paper](https://goo.gl/FlcsMg). In addition, the approach uses a custom post-processing developed to merge and polygonize network output. Source code is included in this repository. The code is also hosted in a working Amazon Machine Instance located [here](https://aws.amazon.com/).


1. Preprocessing
* Images were enlarged, to be square, so it would be easier to feed them to CNN. gdalwarp binary was used with cubic interpolation. Size was changed to 512x512 for both 3-band and 8-band images. Usage example:
```shell
gdalwarp -ts 512 512 -r cubic 3band_AOI_1_RIO_img1194.tif 3band_AOI_1_RIO_img1194.resized.tif
```
* Due to limited amount of training data and low resolution of additional bands, it was decided to use only 4 bands: RGB and infrared. Script to produce 4d numpy arrays for DNN input and 2d numpy arrays for DNN output was developed. Expected output is created from geojson. According to problem statement, annotation has quite low quality and building borders are sometimes shifted. So, it was decided to learn only 2 classes: “building”, “not-building”. To distinguish between two close buildings, border between buildings was assigned to “not-building” class. A script was created that depends on gdal, shapely and cv2(opencv) that can be installed with pip and aptitude. Amplitude in the input channels is normalized per image and rescaled to be in [0; 1]. Example of input (channel 1) and expected output (channel 1. Usage:
```shell
create_feats.py --band3 3band_AOI_1_RIO_img1194.resized.tif --band8  8band_AOI_1_RIO_img1194.resized.tif  --geo Geo_AOI_1_RIO_img1194.geojson --in-dir infeats --target-dir targetfeats
```
![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/bic-user/images/image1.png)
* To have more training data, each 512x512 was cropped into 128x128 images with 50% overlap in both dimensions. That results to 49 training samples per image. If at least one sample was black (sum equal to 0), whole image was dropped. Example of cropped image seen below. Script parameters used:
```shell
crop_feats.py --infeat in_AOI_1_RIO_img1194.npy --outdir infeat_croped

crop_feats.py --infeat target_AOI_1_RIO_img1194.npy --outdir outfeat_croped --to-reshape
```
![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/bic-user/images/image2.png)

2. FCNtraining

Model was created using Keras with Tensorflow backend. During training data was reshuffled, each image was flipped left-to-right and up-to-down. Training was done on AWS g2.2xlarge (single GPU with 4 GB memory). One epoch took approx. 3 hrs. Only 2 epochs were done for final model. Mostly because of time constrains. Also, it was observed that further training brings small improvement according to loss. Following model architecture was used:

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/bic-user/images/image3.png)

First convolutional block (2 conv layers and max-pooling) has 32 filters of size 3x3, with maxpooling of 2x2. Second block – 64 filters of same size. Third block – 128 filters of same size. Before deconvolution there are 2 conv layers: 1024x5x5 and 128x1x1. Due to maxpooling, dimension of the images at this point is 16x16. To go back to original dimension, deconvolution with 8x8 subsampling is applied. Reshaping of the output is applied to do the softmax, to classify each pixel into “building”, “not-building”. Training was performed with Adam optimizer and batchsize 16.

All the training parameters and procedure itself is hardcoded in the script:

```shell
create_model.py --indir infeat_croped --targetdir outfeat_croped
```

Script saves final model in current directory. It can be used for prediction given input features only. Input features are generated as following:

```shell
create_feats.py --band3 3band_AOI_1_RIO_img1194.resized.tif --band8 8band_AOI_1_RIO_img1194.resized.tif --in-dir feats4prediction --only-infeats
```
Prediction using obtained model is run as following:
```shell
predict.py --model model_final.bin --image feats4prediction/in_AOI_1_RIO_img1194.npy --outdir predicted
```
This predicts all the segments from given image. For example:

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/bic-user/images/image4.png)

This image should be vectorized.
