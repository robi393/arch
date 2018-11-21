Research papers:
https://projekter.aau.dk/projekter/files/63218113/report.pdf
https://www.irjet.net/archives/V4/i4/IRJET-V4I4275.pdf
http://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
http://www.scitepress.org/Papers/2018/67188/67188.pdf
https://pdfs.semanticscholar.org/5fd7/eca5432e4e4038dfc2ad425fa5f3e8739f0b.pdf
https://ieeexplore.ieee.org/document/7124941
https://e-archivo.uc3m.es/bitstream/handle/10016/7089/traffic_escalera_IVC_2003_ps.pdf
https://arxiv.org/ftp/arxiv/papers/1707/1707.07411.pdf
https://towardsdatascience.com/what-is-the-best-programming-language-for-machine-learning-a745c156d6b7

Haar Cascade
Code example:
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
How training works:
https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html
Blog post tutorial:
https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html
Another tutorial (check it out):
http://christopher5106.github.io/computer/vision/2015/10/19/create-an-object-detector.html

===================================================================================================
How to do the cascade training:
===================================================================================================
Step 1
- Create a folder structure like this:

/img
  img1.jpg
  img2.jpg
bg.txt

File bg.txt:
img/img1.jpg
img/img2.jpg

==========================================
==========================================
  Introduction
  Diagnosting imaging has become a very important tool in medicine. It provides a way to map the anatomy of a patient which then can be used for diagnosis, treatment planning or medical studies. Medical imaging is used in various modalities such as magnetic resonance imaging (MRI), digital mammography, ultrasound or computed tomography (CT). As technology evolves more and more medical images are being used in every day medicine and reliable methods and algorithms are needed to process the large amount of data aquired from the patients. These algorithms often provide a way to delineate regions of interests (anatomical structures, organs).
  
  Definitions
  "Image segmentation is the process of partitioning an image into distinctive regions (segments) with similar properties such as intensity, texture, brightness, and contrast." [x]
  
  Due to the fact that there are numerous image scans that need to be processed manual segmentation would take a long time and it would often leads to errors and this is why automated methods are needed to perform segmentation.
  
[x] https://ieeexplore.ieee.org/document/7881449

https://pdfs.semanticscholar.org/d909/6dc6d76148633f168c929e6e7961f563ca78.pdf

Moreover, due to the numerous image scans for each patient and similar intensity of the nearby organs, manual segmentation is a tedious task and is prone to errors. Therefore, reliable automated algorithms should be utilized to delineate the region of interest.

------------------------------
1.) Vision For Test Drives (->)
Phase I. (2 HC on board)
Bosch as driver
Bosch as engineer
//measurement requires a lot of adjustments, plus tagging needs to be done
Phase II. (2 HC on board)
AMS driver
Bosch as engineer
//measurement requires a lot of adjustments, plus tagging needs to be done)
Phase III. (2 HC on board)
AMS as driver
BEKO/AMS as engineer
//measurements are becoming more stable, some intervention is required, plus tagging needs to be done)
Phase IV. (1 HC on board)
AMS as driver
No engineer
//measurements are stable, driver can start the equipment and then do ~100km of driving. Tagging is done offline by AI.

2)
Computer Vision (CV) has been rapidly evolved in the last decade. Today very sophisticated APIs are publicly accessible for image recognition, which help the engineer to do these tasks quickly and efficiently. 
For test drives evaulation, knowing the environment is beneficial, that’s why an engineer does tagging during the drives. This activity can be replaced by AI, with adequate accuracy.
As many drive data could be available in the measurements, the auto-tagging function can go pure image recognition. (e.g.: tagging based on road turn radius, rain-sensor signal, environment temperature etc.)

3) Pyramid
AI
Neural NetWorks
CNN - widespread deep-learning technology for image recognition
faster r-cnn
(+: shows better accuracy compared to other solutions, detects smaller features well; -: slow)

4) Object Recognition - Implementations
Matlab (Computer vision toolbox)
+ Easy to use, many tutorials are available for object det.
+ Commonly used within Bosch
- Limited options compared to Python
- Relatively slow
Python
+ Faster computiong
+ Commonly used
+ Good object det API
+ Better parallelization
- Portability and coding is more challanging
Google APi
+ ready to use service
- slow, expensive, confidentality issues

5) Object rec. software overview
Input video files from measurement camera -> main scrip -> frame extractor (provides frame of the video for the image recognition) -> TensorFlow API: object det api provides per frame results -> main scrip -> result file contains frame number and detected objects

6) Configuring the AI
Config file is needed for:
number of objects to be detected (tunnel, gantry, …)
max objects per frame
thresholds
training config options
and more

7) Training the AI
Training is done throurg the convenient  „LabelImg”
100 frames gives adaquate image recognition performance
Training a new object takes approx 1 hour 
On one frame more objects can belabelled at the same time

8) Resulkts so far
The system is capable of recognizing the following objects on a highway with high accuracy:
Truck
Gantry
Bridge
Tagging a 1.5 hour-long video takes about 2 hours
If we use more training data we can improve the accuracy and robustness of the modell

9) Example pictures

10) Example file

11) Further improvements
Make Auto-Tagging an integral part of the eval tool or the post-processing 
Tagging can be extended into situation recognition by applying vehicle date from the vehicle-bus
Training the modell to detect objects that are very similar to our currently detected objects (to avoid false detections)
Applying plausabilization to output file for avoiding false positives
Turning frame number into time data

--------
GNSS bevezető - ESA könyve
https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf
--------
Edge









------------------------------------------------------------------------------------------
Example
Throughout this essay I used an abdominal CT scan [source] as an example to show the results of different segmentation techniques. To test the described algorithms I used OpenCV, an open source library for computer vision that provides optimized implementation to many of the introduced methods.

---
Edge-based - additional info
Edge-based segmentation strategies assume that different objects are separated by edges and segmentation is performed by identifying the grey level gradients (this approach can be used for images with multiple color channels as well with slight modifications).

The general model for edge based segmentation consists of image smoothing, local edge detection by determining the magnitude and direction of the gradient, edge correction (methods for edge-correction include edge relaxation, non-maximum surpression, hysteresis thresholding), and grouping detected edges.

--
Region-based

Region-growing
Region based methods are based on the assumption that the neighbors of a pixel share similar features within a region so these methods usually compare one pixel to its neighbors recursively. There are many different region based techniques such as region-growing or split and merge.
	One of the region-growing method is called seed segmentation. The basic idea is that we start with a single pixel called seed pixel and we keep expanding the region by adding pixels that share similar features. This method consits of the following steps:
	1.) Calculate the histogram of the image.
	2.) Smoothen the histogram by applying a Gauss-filter.
	3.) Detect peaks and valleys (local maximum and local minimum values) in the histogram. This step is called peakiness test.
	4.) Create a binary image using the threshold value found in valleys.
	5.) Use connected component algorithm for every binary image to find connected regions.
	The recursive version of the connected component algorithm consists of the following steps, assuming we're working on a binary image and the pixel values are either one or zero: 
	1) Start processing the image from the top left corner.
	2) If the pixel value is "1" and it hasn't been labeled yet then assign a new label to it.
	3) Check every neighbor of the pixel and if the value of the neighbor pixel is one and hasn't been labelled yet, assign the same label that was used in step 2.
	4) The algorithm ends if every pixel whose value is "1" has been processed.
