
### **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

All of the project code is contained in the IPython notebook file `P5.ipynb`.

[//]: # (Image References)
[image1]: ./output_images/hog-visualization.png
[image2]: ./output_images/bin_spatial.png
[image3]: ./output_images/color-histogram.png
[image4]: ./output_images/feature-normalization.png
[image5]: ./output_images/sliding-window.png
[image6]: ./output_images/find-car-heat.png
[image7]: ./output_images/threshold.png
[video1]: ./video.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

##### Writeup / README

###### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

#### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In addition to HOG features I also used spatially binned and color histogram features.

The code for this step is implemented by function `get_hog_features()` contained in the second code cell of the IPython notebook `P5.ipynb`

In code cells 3 and 4 I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image1]

I extract Spatially binned features using function `bin_spatial()` implemented in code cell 5 of the notebook. Code cell 6 contains code for simple visualization of these features:

![alt text][image2]

Color histogram features implementation is in code cell 7 and 8. Here is an example of histograms for a sample image:

![alt text][image3]

The code extracting and combining the three feature sets into a single vector is located in code cell 9 of the python notebook.

##### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters training a linear SVM classifier on a random sample set of 1000 car and 1000 not car images. The exploration code is in code cell 10 of the notebook. I executed multiple times this test searching for the highest validation accuracy.

The best results for color space parameter was given by `YCrCb` and `YUV`. I chose `YCrCb` because of higher consistence between multiple runs with randomly selected training sets.

For HOG parameters I selected 8 `pixels_per_cell` and 2 `cells_per_block` because they corresponds to the size of detectable features in the training images of 64x64 pixels size. For gradient `orientations` the value of 9 give me a little higher accuracy.

With these values I was able to consistently achieve around 0.98 prediction accuracy using `LinearSVC` classifier with default parameters.

##### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Initially I used `sklearn.model_selection.GridSearchCV` class to find the optimal SVM kernel and parameters, as well as to train the classifier. The code for it is located in cell 11 of the python notebook.

First I extracted the features for the full training set. Then after normalization using `StandardScaler` I used `sklearn.model_selection.train_test_split()` function to split the data into train and test data set. Here is a visual example of the effect of features normalization:

![alt text][image4]

Finally I fed the training set together with parameter grid to the `GridSearchCV` to train and find the best estimator. I used the following parameter grid:

`{'kernel':('linear', 'rbf'), 'C':[0.1, 0.5, 1.0]}`

The result of this optimization was:
```
Best estimator:  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
Best parameters:  {'kernel': 'rbf', 'C': 1.0}
Test accuracy of the choosen SVC:  0.9955
```
Although this classifier has very high accuracy and later it was giving very few false positive detections, the performance was in order of magnitude worse than `LinearSVC`. The time for processing a single frame in the final pipeline was more the 20 seconds. That is why I switched back to linear classifier as my final choice. The optimization code in cell 11 is commented out.

#### Sliding Window Search

##### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In code cell 13 I used the window search function from the lessons to test car detection and experimenting with different window sizes and overlapping. Here is are two examples with window size 96x96 and 75% overlapping:

![alt text][image5]

From the experiments on the test images I concluded that window size should be from 64x64 to 128x128. Most of the cars were detected with 96x96 window.

##### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

In order to optimize performance I used subsampling of HOG features. The final search function `find-cars` is implemented in code cell 15. It calculates the HOG values over the whole search region of the image.

Here are some example images of overlapping detections and corresponding heatmaps:

![alt text][image6]

More examples could be found in the notebook file.

---

#### Video Implementation

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video.mp4)


##### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example showing the heatmap thresholding and applying the label function to combine multiple detections:

![alt text][image7]

The function `add_heat()` implemented in code cell 16 stores the current frame heatmap into array `recent_heatmap` which holds last 15 frames heatmaps. The function returns the sum of all recent heatmaps. This sum is used as an input for `label()` function in the video pipeline (code cell 23).

---

#### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I had is the unreliable linear classifier. It generates too many false positives. In my attempts to filter them out I also removed some of the positive detections. I had much better detection using SVM with RBF kernel but its performance was unacceptable. I would much prefer using deep CNN for vehicle detection instead.
