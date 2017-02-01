# Topcoder: [wleite's](https://www.topcoder.com/members/wleite/) implementation

#### Competition: [SpaceNet Challenge 1 - Rio de Janeiro Building Footprint Extraction](http://crowdsourcing.topcoder.com/spacenet)

#### [Final Rank](https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16835): 1st

#### Official [Building Footprint Metric](https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb#.q0v9inh3i) Score: 0.255292

#### Dataset: [SpaceNet AWS S3](https://aws.amazon.com/public-datasets/spacenet/)

#### Approach:
After reading the problem statement and examining the provided images for a while, the problem is composed of the following sub-problems:
* Classify pixels in the image into 3 categories: border, inside a building and others (outside buildings);
* Based on individual pixel classification, generate candidate polygons that may contain buildings;
* Evaluate polygon candidates, in order to select the ones with higher chance to be correct and discard the others;
* Change each polygon a bit (moving, rotating, resizing), reevaluate it (using the same evaluation function mentioned above) and keep the new polygon if it has a better score (higher probability to be correct).

In practice, after trying many different ways of implementing it, we had to abandon the last step (polygon small improvements) of initially considered solution, as it only decreases the score. WE still believe that it should help, if implemented in a better way.

During the final days of the competition we added another way of classify pixels, based on their distance to the border (in fact, regression was used, as it outputs a continuous numeric result). In practice, it didn’t work well, but we decided to keep it, and use it as a feature in the polygon evaluation step. The following sections describe the main steps of the final solution, after some general observations.

Provided Data:

The number of provided training images didn’t seem very high, but after working with them for few days it was clear that it is large enough, which made sense since each image can contain many buildings. We decided to split the given training images into 3 sets: 60% for pixel classification training, 30% for polygon evaluation training and 10% for local tests. During the competition, we usually used a subset of these sets to run quicker tests. It is worth mentioning that we changed the seed (and therefore the way sets were split) couple of times to avoid some bias, but test results showed that sets were big and random enough.

Images Resolution:

Visually inspecting the data, it seems that the resolution of provided images (RGB channels) are not good enough, even for a human, at least for more dense areas (with a lot of small buildings). In the multiband images, this problem is even worse. However, as far as a human can correctly identify most of the buildings with the provided resolution, the program should be able to do the same, although it is not a simple task. In the description, pixels are always taken from RGB images, unless noted otherwise.

Image Preprocessing:

Before working with the images, a simple blur is applied to each of RGB channels, using the standard 3x3 neighborhood of the pixel with the weight matrix ((1 2 1), (2 4 2), (1 2 1)). This blurring step was introduced in a later stage of the contest, and although it seems to lose useful details, the compression
artifacts present in the images are extremely visible, and the applied blurring could help removing these artifacts. In practice, blurring didn’t affect the scores in local tests. After blurring RGB values, each pixel is converted to HSL channels, which will be used later. Finally, a simple edge detection is executed for each channel of the 8-band images and for a gray channel (created from the RGB values). The edge detection combines horizontal and vertical values of edges computed, respectively, by the following matrices: ((1 2 1) (0 0 0) (-1 -2 -1)) and ((1 0 -1) (2 0 -2) (1 0 - 1)).

##### STEP 1 - Pixel Classification Training:

This is the first step of my solution, implemented by the class BuildingDetectorTrainer. As mentioned before it uses as input (up to) 60% of the training images.

It builds two random forests of binary classification trees: one that classifies each pixel as belonging or not to the border, and the other as inside of a building or not. When assigning the ground truth values of each pixel, it uses a 2x2 area, i.e. 4 pixels are analyzed. If two of more of them are “border pixels”, then it is considered as “border”. And the same criterion (2 or more pixels of the 2x2 area) is used for “inside building” assignment. The following figure shows an example image, in which “border” (red) and “inside” (blue) pixels marked.

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/wleite/images/image1.png)

Although our solution uses 2 different binary classification forests, a single classification forest, with 3 different classes could be used. In theory, it would give similar or slightly better results. we just kept this way because we first implemented a binary classification forest for “is building” or not. And after a while added the border classification code.

The third random forests built has regression trees that should output the distance of a pixel from the border of a building. The valid range is from 0 (in the border) to 11. Pixels inside building receive positive values, and outside pixels have negative values, so the actual range is from -11 to 11. The Manhattan distance was used for simplicity, however, in retrospect, we believe that using the regular Euclidian distance would be better. The following figure shows an example image with the assigned distance values: (red to yellow) for positive values, and (blue darker to lighter) for negative values.

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/wleite/images/image2.png)

The resulting random forests were saved into 3 files: rfBorder.dat, rfBuilding.dat and rfDist.dat. My final submission has 60 trees for each random forest. This number was chosen because we used a 64-core machine during the last days of the competition, and each tree is built by a single thread. Using a higher number of trees won’t help much here.

The features used were the same for the 3 forests, and although I tried different features, I didn’t have time to fully explore the possibilities here, including discarding some useless/misleading features. My final submission used 96 features [5 regions * (4 “channels” * 3 “stats features” + 2 “channels” * 2 “texture features”) + 8 channels * 2 features]:


* For region sizes around the pixel of 2x2, 4x4, 8x8, 12x12, and 16x16:
  * For values of H(hue),S(saturation),L(lightness)andE(edge),takenfrom3-band
RGB images:
    * Average;
    * Variance;
    * Skewness.
  * For values of H and E,takenfrom3-bandRGBimages:
    * Two version of binary local patterns.
* For each channel of the 8-band images:
  * Raw pixel value;
  * Edge value.

To keep the number of sampled pixels in a reasonable limit, not every pixel is used for training (that would be a huge number). One of every 5 pixels horizontally and vertically is sampled, so about 1/25 of pixels are used, but as their neighborhood is also taken into account, in fact each pixel is used many times.

To reduce the amount of data, for images with no buildings at all, the distance between samples is increased by a factor of 4.

The memory used by these samples grows O(n) and the processing time, mostly taken by the tree building, grows approximately as O(n log n). I believe that increasing the sampling ratio would give a very small improvement in final results.

##### STEP 2 – Polygon Evaluation Training:

This is the second main step of the solution, implemented by the class PolygonMatcherTrainer. It uses as input (up to) 30% of the training images (a different set of images from the previous step).

Initially, it uses the classifiers previously built to assign for each pixel a probability value of being a “border” and of being “inside a building”. As the classification applies to a 2x2 region (4 pixels), the result is combined with the (up to) four predictions that contain each pixel. The following figure shows the result of this process:

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/wleite/images/image3.png)

As it may be observed, the classification results seem “fine”, although the border detection is not very sharp. The following step is finding polygon candidates from the pixels, based on their classifications (border / inside). There are many different ways of doing that. We picked a simple one, just as a starting point, but due to lack of time, had to keep it, with few tweaks. A better way of finding these polygon candidates would give a huge increase on final score.

What this solution does is first combine border and inside information in a single value, subtracting (possibly weighted) “border” values from “inside” values. The goal is to keep each building region separated from its neighbor buildings, so a simple flood fill can detect a single building. The actual implementation combines values from a 3x3 pixel neighborhood, with the central pixel contributing with a double weight. In the left side of Figure 4 it is shown in blue the values of the subtraction between inside probability and border probability (weight 1.0).


![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/wleite/images/image4.png)

Then a simple flood fill (expanding groups of 4-connected pixels) is executed, using as a parameter a threshold level to decide where to stop flooding (higher values break the result into many smaller groups, while lower values join pixels into a smaller number of larger groups). The right side of Figure 4 shows in green groups detected using a threshold value of 0.3. Finally, a “border” is added to found groups (default value is 5 pixels vertically and horizontally, and 4 pixels diagonally). This is necessary because the building borders, ideally, were not filled. This process is repeated for many threshold values (from 0.09 to 0.60, with steps of 0.03). Figure 5 shows the results for the same image using thresholds 0.2 and 0.4. Also, two different border weights (final submission uses 1.0 and 0.5) and two different border increase sizes (5/4 and 4/3) are used (combined with each threshold level).

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/wleite/images/image5.png)

A convex hull procedure is applied to generate a polygon from a group of pixels. Figure 6 shows the resulting polygon candidates, for the same example image, with parameters used in Figure 4. In the figure is possible to note the 5/4-pixel border added to the group of pixels. We tried to implement a simple “concave hull”, but didn’t have time to make it work as I would like to. Probably this could improve the solution, as there are many concave buildings, and the “extra” area added may confuse the classification algorithm. At this point, the only check made is that there are at least 4 pixels in the connected group. No other verification (intersection, size, shape) is made, as these are only “candidates”.

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/wleite/images/image6.png)

Now the actually training part starts. Each candidate polygon is compared to ground truth buildings, calculating the best IOU value. A random forest of regression trees is built, to predict, for a given polygon candidate, which would be its IOU. Final submission used 60 trees.

A total of 227 features used are:
* General polygon properties:
  * Area;
  * The two side lengths of the smallest rectangle that contains the polygon(testing many rotation angles);
  * Proportion between sides of that rectangle;
  * Proportion between the area of the bounding rectangle and the actual area;
* Predicted “border” and “inside” values:
  * Total sum of “border” values;
  * Total sum of “inside” values;
  * Average“border”values(totalsum/numberofpixels); o Average“inside”values;
  * Ratiobetween“border”and“inside”totalvalues;
  * Differencebetween“border”and“inside”totalvalues;
* For each of these 6 regions {(pixels in the border), (inside the polygon and 1-2 pixels from the border), (outside the polygon and 1-2 pixels from the border), (inside the polygon and 3-4 pixels from the border), (outside the polygon and 3-4 pixels from the border), (other pixels inside the polygon)}:
  * Values of {(“inside” prediction), (“border” prediction), (difference between them), (distance from the border predicted using the RF build for this purpose)}:
    * Average;
    * Variance;
    * Total (a simple sum of all values);
  * For 3-band images, values of {(edge), (hue), (saturation), (lightness)}:
    * Average;
    * Variance;
  * For 8-band images, values of {(edge), (raw pixel value)}:
    * Average.

In sum, a lot of features were used, using pixels inside and near the polygon border, and predictions results obtained from the trained classifiers in these regions. Not sure which are useful in practice, and it should be room for improvement here.

The resulting random forest is saved as a file named rfPolyMatch.dat.

##### STEP 3 – Testing (Find polygons):

For local tests, we used the remaining 10% of training data to evaluate my solution. This is step is very straightforward, as it reuses methods described before. It is implemented by the BuildingDetectorTester class. Using the same process for finding polygon candidates (step 2), a lot of candidates are found and then evaluated using the trained random forest. At this point each polygon candidate has a predicted IOU value. All candidates are then sorted, from higher to lower IOU. Then each polygon is checked to verify if the intersection area with any previously accepted polygon is higher than 27% of either polygon area, and in such case, it is discarded.

Only polygons with a IOU higher than 0.35 are accepted. This number came from local tests, trying to balance precision and recall to achieve the maximal F-Score. In our tests, we observed that final solution was not very sensitive to this choice, i.e. picking other threshold would produce different recall and precision values, but similar F-scores.

##### Advantages and disadvantages of the approach chosen:

It is a straightforward approach, and relatively simple to understand/improve. Not using external code/libraries/models allowed having full control what is going on, although I obviously could overlook some important parts of the solution, since the time available to implement everything from the scratch was limited.

##### Potential improvements to the algorithm:

* Local improvement of polygon candidates, as mentioned before, didn’t work, but should be something useful, if implemented in a better way.
* Building alignment, street/neighbor information, and other more “global context” features were not explored. When we try to mentally mark the buildings in a testing image, it is natural to use “scene” information. Need to find a good (and simple) way of using it.
* Better candidate polygons finding process.
* Improve the distance from the border predictor. Use it as an alternative way for finding polygons candidates.
