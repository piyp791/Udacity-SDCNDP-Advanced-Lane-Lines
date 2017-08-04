## Project Advanced Lane Finding

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/Selection_016.png "Undistorted chessboard image 1"
[image2]: ./images/Selection_017.png "Undistorted chessboard image 2"
[image3]: ./images/Selection_018.png "Undistorted test image 1"
[image4]: ./images/Selection_019.png "Undistorted test image 2"
[image5]: ./images/Selection_020.png "Perspective transformation image 1"
[image6]: ./images/Selection_021.png "Perspective transformation image 2"
[image7]: ./images/Selection_022.png "Color tranformation Visualization 1"
[image8]: ./images/Selection_023.png "Color tranformation Visualization 2"
[image9]: ./images/Selection_024.png "Absolute Sobel Thresholding"
[image10]: ./images/Selection_025.png "Magnitude Sobel Thresholding"
[image11]: ./images/Selection_026.png "Magnitude Sobel Thresholding"
[image12]: ./images/Selection_027.png "Direction Sobel Thresholding"
[image13]: ./images/Selection_028.png "Direction Sobel Thresholding"
[image14]: ./images/Selection_029.png "Magnitude + Direction Sobel Thresholding"
[image15]: ./images/Selection_030.png "Magnitude +  Direction Sobel Thresholding"
[image16]: ./images/hls_s_1.png "HLS S channel thresholding"
[image17]: ./images/hls_s_2.png "HLS S channel thresholding"
[image18]: ./images/hls_l_1.png "HLS L channel thresholding"
[image19]: ./images/hls_l_2.png "HLS L channel thresholding"
[image20]: ./images/Selection_031.png "LAB B channel thresholding"
[image21]: ./images/Selection_032.png "LAB B channel thresholding"
[image22]: ./images/Selection_033.png "Difference in combined threshold and color threshold"
[image23]: ./images/Selection_034.png "Difference in combined threshold and color threshold"
[image24]: ./images/Selection_035.png "Sliding window visualization 1"
[image25]: ./images/Selection_036.png "Sliding window visualization 2"
[image26]: ./images/Selection_037.png "Sliding window visualization 3"
[image27]: ./images/Selection_038.png "Previous fit plotting visualization 1"
[image28]: ./images/Selection_039.png "Previous fit plotting visualization 2"
[image29]: ./images/Selection_040.png "Lane plotting visualization 1"
[image30]: ./images/Selection_041.png "Lane plotting visualization 2"
[image31]: ./images/Selection_042.png "Lane plotting with data visualization 1"
[image32]: ./images/Selection_043.png "Lane plotting with data visualization 2"
[image33]: ./examples/color_fit_lines.jpg "Finding land pixels from polynomial fit."

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README


### Files

Code File - Advanced-Lane-Finding.ipynb 
Write up File - writeup.md

The jupyter notebook file has been documented with regards to what functionality each part of code implements.

### Camera Calibration

Before moving onto any further processing on the images received from the camera, it is important that the image from the camera is not distorted. For that, the camera first has to calibrated. 
We use the chessboard images provided to us in the camera_cal folder for the purpose of calbration and undistortion. 

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Chessboard images are used to calibrate the camera. The calibration is done using the object points which are the (x,y,z) coordinates of the chessboard corners in real world and the image points of the detected chessobard corners. This is done using the `findChessboardCorners` function of opencv module. 
On getting the object points and the image points and the object points, we compute the calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the chessboard image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Distortion-correction.

Here is an example of distortion correction, applied on test images.

![alt text][image3]
![alt text][image4]


#### 2. Perspective transformation

I chose to hardcode the source and destination points in the following manner:

```python    
src = np.float32([(575,464),
              (707,464), 
              (258,682), 
              (1049,682)])
dst = np.float32([(450,0),
              (w-450,0),
              (450,h),
              (w-450,h)])

where w is the width of the image and h is the height of the image.
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 450, 0        | 
| 707, 464      | 830, 0      |
| 258, 682     | 450, 720      |
| 1049, 682      | 830, 720        |

Then `getPerspectiveTransform(src, dst)` function of the opencv library is used to get the transform and the inverse transform matrix.
Here's what it looks like:

![alt text][image5]
![alt text][image6]

#### 3. Color Transformation

We try the following color transformations:
1. RGB-HSV
2. RGB-HLS
3. RGB-YUV
4. RGB-LAB

**Transformation Visualization 1**

![alt text][image7]

**Thougts**
The original image has quite a bit of sunlight , hence the lanes are not very dark colored. 
1. RGB does an average job of detecting the lanes. The best job is done by the R channel of RGB.
2. G channel of RGB detects the white lane decently.
3. S channel of HSV does a decent job of detecting yellow lane, but not quite good with the white one. Same with the S channel of HLS
4. L channel of HLS does a decent job at detecting the white pixels.
5. U channel of YUV and B channel of LAB color spaces perform at a similar level st detecting yellow pixels.


**Transformation Visualization 2**

![alt text][image8]

**Thougts**
This image has a lot of noise due to noisy lighting conditions.
1. RGB does a much better job when the lighting conditions are not bright.
2. S channel of HSV and HLS does a decent job of detecting yellow lane, but seems vulnerable to noise.
3. Y channel of YUV and R channel of RGB do a similar job at detecting lane pixels.
4. U channel of YUV and B channel of LAB color spaces perform at a similar level st detecting yellow pixels.


#### 5. Absolute Threshold 
![alt text][image9]

#### 6. Magnitude Threshold
![alt text][image10]
![alt text][image11]

#### 7. Directional Threshold
![alt text][image12]
![alt text][image13]

#### 8. Magnitude + Directional Threshold
![alt text][image14]
![alt text][image15]


**Thoughts on sobel transformation**

While the edge are detected in images by application of sobel transformation, it still leaves a lot to be desired as it doesnt seem to perform that well in noisy lighting conditions.
Maybe color transformation can solve this.

#### 9. HLS S Channel Threshold
![alt text][image16]
![alt text][image17]

#### 10. HLS L Channel Threshold
![alt text][image18]
![alt text][image19]

#### 11. LAB B Channel Threshold
![alt text][image20]
![alt text][image21]


**Thoughts on color transformation**

As we can see from the images above, S channel of HLS channel does a good job at detecting yellow lane pixels, but is vulnerable to noise. Compared to that, B channel of the LAB color space does an equally good job at detecting yellow lane pixels, and loks resistant to noise.

L channel of HLS does well at detecting white lane pixels.


##Connecting the complete pipeline

1. perform distortion correction.
2. perform sobel and color thresholding.

Lets visualize the difference in sobel and color thresholding.

![alt text][image22]
![alt text][image23]

**Thoughts**

Although it depends entirely on the values chosen for the threshold, from what has been chosen currently, it seems sobel thresholding seems to be picking up a lot of noise besides the lane lines as compared to only color thresholding.
Lets go ahead with the color thresholding only and see what we get with that.


#### 4. Polynomial Fitting for lane pixels

Now this was the interesting part, albeit time consuming!

Given this image, how can we go about fitting a polynomial on the lane pixels?

1. Divide the problem into smaller problems by using smaller windows and detecting useful pixels for each window
2. Once a certain number of useful pixels have been detected for that window, move onto next windoe and do tha same.
3. Once useful pixels for all the windows have been detected for both the lanes, fit a polynomial using the function `np.polyfit`.

Note: The base points for the starting window are found using the histogram of the binary image. 
The assumption is that the peaks on the histograms coorespond to the left and right lane pixels. To avoid picking any noise, we find the peaks between a certain fixed boundary.

Lane pixels from the calculated fits are found using the following formula:
![alt text][image33]


#### 4. Visualize the sliding window method

![alt text][image24]
![alt text][image25]
![alt text][image26]

Note that even though the peak resulting due to the noise at the extreme ends of the frame are greater than the lane peaks, still they are not considered due to the our criteria for considering base points for each window of not going to the extreme ends of the frame to look for peaks

#### 6. Polynomial Fitting for lane pixels using Previous Frame information

The next step to use information from the previous frame for fitting polynomial on lane pixels.
We use the left and the right fit obtained from the previous frame and use them for plotting the lane lines on the current frame.

#### 4. Visualize the Prev Plot Fit method

![alt text][image27]
![alt text][image28]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated using the formula mentioned in this [resource](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and [Udacity lesson ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f)

The important point to notice is that radius if curvature is in m units and so far we have been dealing distances in pixels. So we need to come up with a maping between the two.
Apparently, in y dimension, there are approximately 3.048 meters per 100 pixels and 3.7 meters per 378 pixels in the x dimension.

Also, the point where the radius of curvature is calculated has to be decided. We calculate the radius at the base of the image.

The distance from the center of the lane is calculated as the distance between the center of the lane (which is the means of left and right lane intercepts)and the position of the car. (You don't say!)

#### 4. Visualize the radius of curvature

![alt text][image31]
![alt text][image32]


#### 6. Detected Lane plotted on Original image

![alt text][image29]
![alt text][image30]

---

**Final image Processing**

1. Thresholding using the function `pipeline_color`.
2. If polynomial fit present from previous frame, use that information to plot lane lines using the function `polyfit_using_prev_fit`, else use the function `lane_polyfit` to find appropriate fits for lane pixels
3. Validate the fit obtained by calculating the difference between the left and right lanes x intercepts. The ideal distance is 350 pixels. If the difference in calculated left and right lanes is greater than 100 pixels from the 350 px width, discard those fits, else add those fits to the left and right Line class* instances
4. Calculate the radius of curvature and distance from the center.
5. Draw the fitted lane lines and the data onto the original image

**Line Class** - To store information about lane lines such as polynomial fits, best fits, radius of curvature etc, we create a separate class Line , which is instantiated twice, once for each left and right lane lines.


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[![Project video](http://img.youtube.com/vi/_4e6sZbLUa8/0.jpg)](http://www.youtube.com/watch?v=_4e6sZbLUa8 "Lne finding on Project video.")

**Thoughts**

The pipeline words fine on the project video and looks robust on changes on lighting conditions and other noises.

#### 2. Challenge Video

[![Challenge video](http://img.youtube.com/vi/TTAy4qHyE-I/0.jpg)](http://www.youtube.com/watch?v=TTAy4qHyE-I "Lne finding on Challenge video.")


**Thoughts**

This one is a bit less stable. There are stretches where the pipeline breaks down completely, down to the changes in lighting conditions possibly.

#### 3. Harder Challenge Video


[![Harder Challenge video](http://img.youtube.com/vi/QoDoPf-6QpE/0.jpg)](http://www.youtube.com/watch?v=QoDoPf-6QpE "Lne finding on Harder Challenge video.")

**Thoughts**

Well , the pipeline breaks down completely for this one. The changes in lighting are sudden, so are the changes in radius of curvature. 
I wouldnt want to be sitting behind the wheel with this lane finding pipeline.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


The pipeline is likely to (and infact does) fail on the harder challenge video.
The implementation doesnt work that well on the harder challange video. Infact, it doesnt work at all. Diagnosis shows problems with the :

**1. Region of interest coordinates calculation doesnt work well enough.**

The implementation for calculating region of interest could be made more robust by moving away from hard coding and coming up with a calculation scheme that takes into account the parameters of the image received such as center, width , height more effectively.
    
**2. Thresholding doesnt work that well either.**

Abrupt changes in light conditions breaks the color thresholding and a lot of noise is picked up from the surroundings. Coming up with better threshold values can make it more robust. 

**3. Sudden changes in radius of curvature throws off the lane finding pipeline.**

To make it more robust:
1. We could try fiddling with the threshold values a bit more.
2. We could try improving the implementation for finding the region of interest.
3. We could try to make polynomial fitting implementation more robust by accomodating for scenarios where the changes in radius of curvature are sudden.





