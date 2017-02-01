## Topcoder: [fugusuki's](https://www.topcoder.com/members/fugusuki/) implementation

#### Competition: [SpaceNet Challenge 1 - Rio de Janeiro Building Footprint Extraction](http://crowdsourcing.topcoder.com/spacenet)

#### [Final Rank](https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16835): 4th

#### Official [Building Footprint Metric](https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb#.q0v9inh3i) Score: 0.216199

#### Dataset: [SpaceNet AWS S3](https://aws.amazon.com/public-datasets/spacenet/)

#### Approach:
This approach consists of 3 parts listed below.
1. Probability of being inside and edge of buildings of each pixel are
predicted by Fully Convolutional Network (FCN).
2. Building polygons are created from probability heat map by using
greedy algorithm for cluster growing.
3. Confidences of being a building for each polygon are predicted by
Gradient Boosting Decision Tree (GBDT) and polygons with less
confidence are removed.

Part 3 is a characteristic part of this method. Precision was significantly
improved by using part 3. Part 1 may not be so different from existing or other competitors’ methods. Recall was not so good compared with precision. Part 2 may be too simple and primitive. Source code is included in this repository. The code is also hosted in a working Amazon Machine Instance located [here](https://aws.amazon.com/).

#### Detailed description of the algorithm
##### Part 1: Probability prediction by FCN (train[line:138] for training / pred_proc [line:354] for prediction)

A FCN model is used for predicting probability heat map of being inside of building. Keras library was used. properties of the model are listed below.

Input data: The size was 408(height) x 440(width) x 11(channel). First 3 channels
were pixels of 3band image and next 8 channels were pixels of enlarged
8band image.

Label / Output: The size was 408(height) x 440(width) x 2(channel). For channel 1,
pixels inside of building were 1 and outside were 0. For channel 2,
boundary pixels of building were 1 and the others were 0.

Structure of FCN: Input layer and output layer were connected with several dozens
of convolution, pooling, upsampling layers. Batch normalization was applied for each layer and LeakyReLU activation was used for every hidden layer because recently these are widely used.

Hyper parameters: binary_crossentropy was used as loss function and Adam algorithm was used as optimizer. Training iteration was About 100 epochs x 1000 samples.

Please view CreateData function [line:47] for detailed process of creating input data and label. Also, create_model function [line:78] for structure of FCN and hyper parameters.

##### Part 2: Creating building polygons (FindAllClusters [line:225])

The idea was basically same as greedy algorithm for cluster growing that is described in the “Post-processing” section in the article “[Object Detection on SpaceNet](https://medium.com/the-downlinq/object-detection-on-spacenet-5e691961d257)” in the Asset Library of this competition.  This method can be improved as said in the article but I could not do it except modification of some thresholds. During this process, features of the polygons that will be used in next part are extracted.

##### Part 3: Confidence prediction by GBDT (FindAllClusters [line:225] for feature extraction, train_building [line:426] for training, test_proc [line:461] for prediction)

Polygons created by previous part have features such as cluster size, intensity distribution in the cluster, cluster size growing history and so on. For polygons created from the training images, IOU scores can be evaluated. By learning relevance between the features as input features and IOU score as label, model for predicting IOU score can be created. XGBoost library was used. Please view FindAllClusters function [line:225] for details about input features and train_building function [line:426] for detailed parameters.

Dropping polygons with lower predicted IOU score (i.e. lower confidence for being building) improved the precision significantly while recall was not improved. Provisional score was improved by about 10000.
