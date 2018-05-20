# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Projec**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/training_set_bar_graph.png "Visualization"
[image2]: ./writeup_images/sign_before_normal.png "Before Normalization"
[image3]: ./writeup_images/sign_after_normal.png "After Normalization"
[image4]: ./test/images?q=tbn:ANd9GcTSzZj1ZMmeVFCnaRi5B3p_aW3NF5kYkskYRrkF0IcL175cJqkX3Q "Traffic Sign 1"
[image5]: ./test/30-mph-sign-mikecogh.jpg?h=468 "Traffic Sign 2"
[image6]: ./test/100_1607_small.jpg "Traffic Sign 3"
[image7]: ./test/64914157-german-road-sign-slippery-road.jpg "Traffic Sign 4"
[image8]: ./test/images?q=tbn:ANd9GcSF1z49fQZWTqzOD2IXgs2ZheMMqA6UNALlGyHGvxFI96noXtVW "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bhiln/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python APIs to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between examples of all the 43 different traffic signs. As one can see, there are many examples of class 2 and 3, while not many of 1 and 20. This could negatively affect the model by underfitting 1 and 20, while possibly overfitting 2 and 3

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the images to be between -1 and 1, centered around 0. This makes the calculations done by Python more accurate, as they are smaller numbers. I decided not to grayscale the images because this will remove some important information that the model can use to classify images. For example the red would be removed from a stop sign, making the model rely on shape. This does make the model more complex and take longer to train, but at this scale, that's alright to leverage to give more accurate results.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2] ![alt text][image3]

For this project, I didn't add any extra pre-processing. However, I have read and learned about other techniques and understand the advantages:

1. Rotation: duplicating, rotating n degrees, and then appending an image (for every image and many different n degrees per image) can significantly grow the training database. Without having to collect more images, this adds data that relates to real life scenarios, as signs, for example, can be rotated at any angle. This will allow the model to recognize new images even if they are rotated.
2. Adding noise to images. This gives more information to the model to be able to overcome grainy images it might encounter after being trained
3. Translating the image to many different locations can again grow the number of unique images in the training database. This can help the model learn that a stop sign is still a stop sign, no matter where it is on the page. This can help in addition to pooling.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU 					| 												|
| Max pooling 			| 2x2 stride, outputs 5x5x16					|
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU 					|												|
| Fully connected		| outputs 84									|
| RELU 					|												|
| Fully connected		| outputs 43									|
| Softmax				| 	        									|
| Loss					|												|
| Adam Optimizer		|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer. I played around with different amounts of epochs. In the end, I decided to use a variable amount, depending on the accuracy. This is described in #4 below. I also played around with the batch size. Originally it was set to 128. However, I know that, if I have the recourses, smaller batch size is probably better, as it doesn't generalize as much. I found that a batch size of 32 worked well for me. 64 probably would have been fine as well. In addition, I used a variable learning rate, as also described below in #4.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First I tried 20 epochs with a learning rate of 0.001. Then I tried 10 epochs with a learning rate of 0.002. I found these to be somewhat comparable. But what I noticed is that it was highly inconsistent. Sometimes I would hit the goal of 0.93 accuracy, and sometimes I wouldn't. I decided to try setting a "goal", which monitors the accuracy and stops the training once the training accuracy surpasses the set goal, with a maximum number of epochs set so it won't run forever. Since we are aiming for 0.93 accuracy, thats the goal that I set. I felt that this would be efficient use of the model, instead of training for 10 and only getting to 0.927, when one more epoch could have gotten over 0.93, for example. I set the maximum number of epochs to 300. With this approach, I set the starting learning rate to 0.001. I realized that as I hit certain "thresholds" of accuracy, it would probably be a good idea to lower the learning rate as to not overshoot the local min. My reasoning is that it might take longer to reach goal, but if it overshoots or goes the wrong way, it won't be by that much. This idea is similar to trickle charging a device. In my model, once I hit accuracy of 0.9, I changed the learning rate to 0.0009. Then when accuracy is 0.91, learning rate is 0.0008. And finally, when accuracy is 0.92, learning rate is 0.0005. This seemed to work pretty well.

My final model results were:
* validation set accuracy of 0.934
* test set accuracy of 0.912

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
** I stuck with the LeNet architecture.
* What were some problems with the initial architecture?
** Adding dropout seemed to make it worse.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
** I tried adding a dropout operation, however this seemed to make my accuracy worse, which is to be expected at first. As it ran, though, the accuracy would improve, but not reach the goal I was expecting. I ended up taking dropout out. But my test accuracy was only 0.91. This probably could have been improved by making the model more generic using dropout.
* Which parameters were tuned? How were they adjusted and why?
** I tuned learning rate, batch size, and number of epochs. My reasoning is described above in #4.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
** Convolution layer is good for image recognition because all images are different. It is not guaranteed that images will have the same area of interest, which might be translated. Convolution can help with that by adjusting the shape of the input to normalize the image. Dropout can improve the model because it will ensure that the model does not "memorize" the training data. It will be successful for the training data, but also generic enough to be able to work on any other future input it is given, even if it is new data.

If a well known architecture was chosen:
* What architecture was chosen?
** LeNet architecture was chosen because it was suggested and worked for me.
* Why did you believe it would be relevant to the traffic sign application?
** Different NN architectures work better for different applications. It is well known that LeNet works well for images recognition and classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
** It is evident that the model is working well because it is accurate on both the training and test sets. If it worked well for the training but not the test set, then the model has "memorized" the test set, which is not good. In that case, number of epochs will have to be decreased or dropout layers will have to be added.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Here are the results of the prediction:

| Image			    				    |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Children crossing						| Children crossing								| 
| Speed limit (30km/h) 					| Speed limit (30km/h) 							|
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Slippery road	     					| Slippery road					 				|
| Road work								| Road work      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.912.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a children crossing sign (probability of 0.96). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.95671763  			| Children crossing								| 
| 0.04308151			| Dangerous curve to the right					|
| 0.00020063			| Beware of ice/snow							|
| 0.00000022   			| End of no passing				 				|
| 0					    | Slippery Road      							|


For the second image, the model is somewhat sure that this is a Speed limit (30km/h) sign (probability of 0.75). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.7471346098016274 	| Speed limit (30km/h) 							|
| 0.25261194308905965 	| Pedestrians 									|
| 0.00012429874083516847| Vehicles over 3.5 metric tons prohibited 		|
| 8.79549928200201e-05 	| General caution 								|
| 4.1193375657598544e-05| Right-of-way at the next intersection 		|

For the third image, the model is completely sure that this is a Right-of-way at the next intersection sign (probability of 0.99). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.999999999436719 	| Right-of-way at the next intersection 		|
| 5.632806506757398e-10 | Pedestrians 									|
| 3.8098677122494636e-16| Road narrows on the right 					|
| 2.481191874581544e-20 | Traffic signals 								|
| 2.8870226186507615e-22| Beware of ice/snow 							|

For the fourth image, the model is completely sure that this is a Slippery road sign (probability of 0.99). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.9999993956503195 	| Slippery road 								|
| 6.039955875918068e-07 | Road narrows on the right 					|
| 3.54092879991963e-10 	| Dangerous curve to the right 					|
| 4.881096218334195e-18 | Beware of ice/snow 							|
| 1.9836416836300783e-18| Children crossing 							|

For the fifth image, the model is completely sure that this is a Road work sign (probability of 0.99). The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.9999999999991215 	| Road work 									|
| 8.623413456474935e-13 | Priority road 								|
| 1.624232589980499e-14 | Bicycles crossing 							|
| 1.5381384459303215e-18| Beware of ice/snow 							|
| 9.429385459595892e-22 | Wild animals crossing 						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


