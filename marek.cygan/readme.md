## Topcoder: [marek.cygan's](https://www.topcoder.com/members/marek.cygan/) implementation

#### Competition: [SpaceNet Challenge 1 - Rio de Janeiro Building Footprint Extraction](http://crowdsourcing.topcoder.com/spacenet)

#### [Final Rank](https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16835): 2nd

#### Official [Building Footprint Metric](https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb#.q0v9inh3i) Score: 0.245420

#### Dataset: [SpaceNet AWS S3](https://aws.amazon.com/public-datasets/spacenet/)

#### Approach:
##### What to predict?

The two options that were considered to create polygons with this model were:
* make pixelwise predictions deciding whether a pixel belongs to a building footprint or not, and then create arbitrary polygon shapes out of the pixels,
* predefine a set of polygons which are possible prediction outcomes and use only this set.

This implementation takes the second option drawing inspiration from the article by Liu et al. [2]. Source code is included in this repository. The code is also hosted in a working Amazon Machine Instance located [here](https://aws.amazon.com/).

##### Computational pipeline
First, we try to classify pixels into 3 categories - inside, outside and border, where the border is defined as everything which is inside and at most 4 pixels away from the building boundary as shown below.

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/marek.cygan/images/image1.png)

Next, given the original image and the predicted heatmap we split into a 50×50 grid, where in each of the 2500 pieces we have a set of default 16 rectangles (called default boxes in [2]) depicted in Fig. 2. For each of the rectangles we want to predict whether this default rectangle has an IoU with some building footprint above the threshold of 0.5. If it does, similarly as in [2] we have a regressor for changing its width, height, translation. The one additional regressor comparing to [2] is angle - in [2] all the rectangles are axis parallel and using a regressor we allow rotation.

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/marek.cygan/images/image2.png)

##### Architecture
For predicting the heatmap, we build the following architecture using dilated convolutions [3]. All convolutions are 3 × 3 and do not change the size of the image which remains 400 × 400 throughout the process.

1. input was rescaled image of size 400 convolution with 16 filters, dilation 2, × 400 with 11 channels (all bands),
2. convolution with 16 filters, dilation 1,
3. convolution with 16 filters, dilation 1,
4. convolution with 16 filters, dilation 2,
5. convolution with 16 filters, dilation 4,
6. convolution with 16 filters, dilation 8,
7. convolution with 16 filters, dilation 16,
8. convolution with 16 filters, dilation 32,
9. convolution with 16 filters, (being the logits for the 3 classes of pixels), dilation 16.

Each convolution was followed by batch normalization layer [1].

For the rectangles, the architecture (all convolutions are 3 × 3 with padding ”SAME”, max pool has stride 2):
1. input was a rescaled image of size 400 × 400 with 3 channels (tried using all 11 bands but this did not help),
2. convolution with 20 filters,
3. convolution with 20 filters, max pool,
4. convolution with 40 filters,
5. convolution with 40 filters, max pool,
6. convolution with 80 filters,
7. convolution with 80 filters,
8. convolution with 80 filters, max pool,
9. convolution with 160 filters,
10. convolution with 16 · 6 filters (for each of the 16 default rectangles there is a logit and 5 regressors for scale, offset and rotation).

Each convolution was followed by batch normalization layer [1].

##### Training Process
Training the heatmap model:
```shell
python -u main.py --rectangles 144 --pieces 50 --batch size 8 --lr 0.001 --dilarch 2 --num epochs 93 --mode heatmap --final bn 1
```
The final model is then stored in the models/epoch93.ckpt file.
Training the rectangles model:
```shell
python -u main.py --rectangles 16 --pieces 50 --batch size 16 --lr
0.0002 --epoch decay 1.0 --mode rectangles --num epochs 32 --arch multiplier 5 --mse mult 10
```
The final model is then stored in the models/epoch32.ckpt file.

Batch size and learning rate (lr) can be found in the command line arguments above. ADAM was used as the optimizer.

##### Predictions
In order to make predictions, one needs to run ./clean.sh, then ./compile.sh, followed by ./run.sh train-data-folder test-data-folder output-file.

##### References

[1] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Francis R. Bach and David M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 448–456. JMLR.org, 2015.

[2] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott E. Reed, Cheng-Yang Fu, and Alexander C. Berg. SSD: single shot multibox detector. CoRR, abs/1512.02325, 2015.

[3] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. CoRR, abs/1511.07122, 2015.
