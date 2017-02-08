# Topcoder: [qinhaifang's](https://www.topcoder.com/members/qinhaifang/) implementation

#### Competition: [SpaceNet Challenge 1 - Rio de Janeiro Building Footprint Extraction](http://crowdsourcing.topcoder.com/spacenet)

#### [Final Rank](https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=16835): 3rd

#### Official [Building Footprint Metric](https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb#.q0v9inh3i) Score: 0.227852

#### Dataset: [SpaceNet AWS S3](https://aws.amazon.com/public-datasets/spacenet/)

#### Approach:
In the SpaceNet challenge, competitors are asked to develop automated methods for localizing every building footprint in pixel level from high-resolution satellite imagery. Therefore, this task can be regarded as instance segmentation in the area of computer vision. Fortunately, based on my experience and recent work on instance segmentation, I modified the Multi-task Network Cascades(MNC)[1] as our tool to settle this problem. The traditional MNC have three stages, each of which addresses one sub-task. The cascade respectively includes proposing box-level instance, regressing mask-level instances, and categorizing each instance. The three stages share their features(e.g., the 13 convolutional layers in VGG-16[2]), as in traditional multi-task learning[3]. Nevertheless, SpaceNet challenge dataset only contains two category, background and building footprint. There is no need to categorize object, I deleted the classification sub-task and only used the first two stage. The method described is the foundation of my submissions to the SpaceNet challenge.

The submitted cascade model can thus be trained end-to-end via a clean, single-step framework. This single-step training algorithm naturally produces convolutional features that are shared among the two sub-tasks, which are beneficial to both accuracy and speed. I use RPN framework[4] to generate region of interest and mask segmentation to obtain the final pixel-level scores for building footprint. My algorithm framework has been showed in Figure 1. However, there are too many building footprints in an image to lead to the GPU out of memory, I have to limit the max sample number to building footprint. It may decrease the benchmark of our algorithm. Therefore, the potential improvement is the use of higher resolution image and residual network. I try to amplify the image 4 times as the input of our network with the help of bilinear interpolation. What a pity, I failed to submit the result score owing to deadline. I firmly believe that this method will greatly improve the benchmark. In addition, I do not use the 8-channel dataset.

##### Data Preprocessing:

The format of data label is Geojson. I preprocessed them into format of mat using jsonlab tools in matlab. Due to this, I follow the instance segmentation work proposed by jifeng dai .at all, I preprocess the data into the format included three folder:inst,cls,img. folder named inst is used for record the buildings appeared in a image. folder named cls is used for record every class in the image.

##### Training Period
I used the ImageNet pretrained models (e.g., VGG-16) to initialize the shared convolutional layers and the corresponding 4096-d fc layers. The extra layers are initialized randomly as in [5]. I adopt an image-centric training framework [4]: the shared convolutional layers are computed on the entire image, while the RoIs are randomly sampled for computing loss func- tions. In our system, each mini-batch involves 1 image, 256 sampled anchors for stage 1, and 64 sampled RoIs for stages 2. I train the model using a learning rate of 0.001 for 38k iterations. I trained the model just in a TITAN. The images are resized such that the shorter side has 600 pixels.

![](https://github.com/SpaceNetChallenge/BuildingDetectors/blob/master/qinhaifang/images/image1.png)


##### Inference Period
The inference process gives a list of 300 instances with masks. I postprocess this list to reduce similar predictions. I first apply NMS (using box-level IoU 0.3) on the list of 300 instances based on their category scores. After that, for each not-suppressed instance, I find its ’similar’ instances which are defined as the suppressed instances that overlap with it by IoU ≥ 0.5. The prediction masks of the not-suppressed instance and its similar instances are merged together by weighted averaging, pixel-by-pixel, using the classification scores as their averaging weights. This armask votingas scheme is inspired by the box voting in [6]. The averaged masks, taking continuous values in [0, 1], are binarized to form the final output masks. One image can be inferred in almost 0.17s.

##### Libraries and Open Source

I modified the MNC code from this [github repo](https://github.com/daijifeng001/MNC). Consequently, I also need the Caffe, a typical deep learning framework. Perhaps some python libaries are also needed such as numpy and scikit. You can configure the environment following the MNC install instruction.

##### Local Programs Used
I have two solutions to train the perfect model. One solution do not need the pretrain model but another does. Fortunately, both solutions can gain the final test result.

No pretrain model

I initialized the network weights with gaussian distribution setting mean 0 and std 0.1. And I trained the model with 60W iterations in almost 5 days. It took a lot of time so that I do not recommend this solution.

Pretrain model

I only need the VGG16M ̇askpretrainedmodeltoinitializeournetwork,which are trained from PASCAL VOC 2012 [dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). This dataset includes twenty object classes like person, animal and vehicle, which are greatly different from the SpaceNet dataset. Therefore, you can train the network successfully even without the pretrained model in considerable training iterations but it may take a lot of time. The network with pretrain model only need 10W iteration. The VGG16Mask pretrained model can be downloaded from [Onedrive](https://onedrive.live.com/download?resid=F371D9563727B96F!91967&authkey=!AKjrYZBFAfb6JBQ). What’s more, the explicit parameters setting are stored in train.prototxt, solver.prototxt, and mnc_confing.py.

References

[1] Jifeng Dai, Kaiming He, and Jian Sun. Instance-aware semantic segmentation via multi- task network cascades. arXiv preprint arXiv:1512.04412, 2015.

[2] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for largescale image recognition. Computer Science, 2014.

[3] Multitask Learning MTL. Multi-task learning.

[4] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE International Conference on Computer Vision, pages 1440–1448, 2015.

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision, pages 1026–1034, 2015.

[6] Spyros Gidaris and Nikos Komodakis. Object detection via a multi-region and semantic segmentation-aware cnn model. In Proceedings of the IEEE International Conference on Computer Vision, pages 1134–1142, 2015.
