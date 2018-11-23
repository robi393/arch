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
Throughout this essay I used an abdominal CT scan [source] as an example to show the results of different segmentation techniques. To test the described algorithms I used OpenCV, an open source library for computer vision that provides optimized implementation for many of the introduced methods.

---
Edge-based - additional info
Edge-based segmentation strategies assume that different objects are separated by edges and segmentation is performed by identifying the grey level gradients (this approach can be used for images with multiple color channels as well with slight modifications).

The general model for edge based segmentation consists of image smoothing, local edge detection by determining the magnitude and direction of the gradient, edge correction (methods for edge-correction include edge relaxation, non-maximum surpression, hysteresis thresholding), and grouping detected edges.

--
Region-based

Region-growing
Region based methods are based on the assumption that the neighbors of a pixel share similar features within a region so these methods usually compare one pixel to its neighbors recursively. There are many different region based techniques such as region-growing or split and merge.
	
	Seed Segmentation
	One of the region-growing method is called seed segmentation. The basic idea is that we start with a single pixel called seed pixel and we keep expanding the region by adding pixels that share similar features. This method consits of the following steps:
	1.) Calculate the histogram of the image.
	2.) Smooth the histogram by applying a Gauss-filter.
	3.) Detect peaks and valleys (local maximum and local minimum values) in the histogram. This step is called peakiness test.
	4.) Create binary images using the threshold value found in valleys.
	5.) Use connected component algorithm for every binary image to find connected regions.
	The recursive version of the connected component algorithm consists of the following steps, assuming we're working on a binary image and the pixel values are either one or zero: 
	1) Start processing the image from the top left corner.
	2) If the pixel value is "1" and it hasn't been labeled yet then assign a new label to it.
	3) Check every neighbor of the pixel and if the value of the neighboring pixel is "1" and it hasn't been labelled yet, assign the same label that was used in step 2.
	4) The algorithm ends if every pixel whose value is "1" are processed.
	
	Split & Merge
	Split and Merge first splits non-homogeneous regions into four equal size regions. After each split a test is needed to determine whether each new region is homogeneous or not. If they're not, they need to be splitted again. This step is repeated until there are only homogeneous regions left. The method is using a quadtree data structure, where total regions are the parent nodes and the splits of the region are the child nodes. The next step is merging neighboring regions that share similar properties. The algorithm stops when we can't find any more regions to merge. Figure X shows an example of this segmentation method.
	
	[http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/medium/segment/split.htm]

Additional sources:
http://www.di.univr.it/documenti/OccorrenzaIns/matdid/matdid125113.pdf
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/region.pdf

A Split & Merge implementation
https://stackoverflow.com/questions/7050164/image-segmentation-split-and-merge-quadtrees/14730467#14730467

---
A Paper
https://ieeexplore.ieee.org/document/7881449
SECTION I.
Introduction

The increase in the use of imaging modalities (such as CT and MR images) for clinical purposes necessitates the use of powerful computers to assist radiological experts for diagnosis and treatment planning. Moreover, due to the numerous image scans for each patient and similar intensity of the nearby organs, manual segmentation is a tedious task and is prone to errors. Therefore, reliable automated algorithms should be utilized to delineate the region of interest.

Image segmentation is the process of partitioning an image into distinctive regions (segments) with similar properties such as intensity, texture, brightness, and contrast [1]–[2][3]. Medical image segmentation on the other hand is a difficult process in nature. Since image data and model prototype (the a priori description of features to be analyzed) are typically complex, usual methods utilized in image processing cannot be directly used in medical image analysis. Moreover, in most medical segmentation application, such as tumor segmentation, regions to be segmented are often non-rigid, vary in size and location, and differ from patient to patient [4]. This makes the segmentation process even more complex and accurate and reproducible segmentation is not straightforward [5].

Also segmentation is a key step of most medical image analysis tasks, such as registration, labeling, and motion tracking. A primary example is the segmentation of the heart, especially the left ventricle (LV), from cardiac images. Segmentation of LV is compulsory for computing diagnostic information such as ventricular volume ratio, heart output, and for wall motion analysis which provides information on wall thickening, etc. [6]. Development and implementation of the related segmentation techniques requires detailed understanding of the underlying problems as well as the nature of the data, and scientific or medical interest.

Clinical acceptance of segmentation methods depends on their computational simplicity and the degree of user supervision [7]. Both semiautomatic and fully automatic segmentation methods are proposed and utilized in medical segmentation. This paper presents an overview of the most applied existing segmentation techniques applied after acquisition of the medical images. The following sections are organized as follows: First, section 2 describes manual and automatic segmentation methods. Section 3 briefly gives an overview of supervised and unsupervised segmentation methods. Then, in section 4, current segmentation methods and their challenges and advantages are discussed. Next, evaluation of segmentation methods is presented in section 5. Finally, the paper is concluded and summarized in section 6.
SECTION II.
Manual and Automated Segmentation

Image segmentation methods can be divided into three groups based on the degree of human involvement: manual segmentation, semi-automatic segmentation, and fully automatic segmentation [7], [8]. These groups are explained in the following section.
A. Manual Segmentation

Manual segmentation of an organ is manually painting the organ or drawing the boundaries of the organ which is done by an expert (physician, trained technician, etc.) [7], [9]. Manual segmentation is a tedious and a time consuming task. It also has “inter-rater” (different experts segment the image differently), and “intra-rater” variability (the same person segments the image differently in different times); hence, it is not reproducible [10]. Since manual segmentation serves as the ground truth for other segmentation algorithms (automatic and semi-automatic), it should count for inter and intra-rater variability. Although manual segmentation depends on the rater, it is still widely used in clinical trials where time constraint is not of importance [11], [12].
B. Semi-Automatic/Interactive Segmentation

In Semi automatic segmentation (SAS) methods, having the minimum human interactions is required to initialize the method or correct the segmentation results manually [9]. SAS methods use different strategies to combine human knowledge and computers. There are three different types of work flow which are often implemented as reported by Ramkumar et.al. [9]. One should keep in mind that SAS methods are subjected to inter-rater and intra-rater variability.
C. Fully Automatic Segmentation

In fully automatic methods, the computer does the segmentation without human interaction. These methods incorporate human knowledge as a priori and usually use model-based techniques along with soft computing methods. In automatic segmentation methods, the use of anatomical knowledge such as size, shape, and location is essential for having a robust algorithm. Since humans use high level visual processing and specialized knowledge to perform segmentation tasks, developing a fully automated algorithm with high accuracy is challenging.

Currently, the fully automatic segmentation methods don't have wide acceptance in clinical practice, while they are advantageous in processing a large batch of images mostly for the research community.
SECTION III.
Supervised and Unsupervised Segmentation

Image segmentation methods are classified according to the employed learning method into three categories: supervised, semi-supervised, and unsupervised segmentation. In supervised methods the full training data is labeled but in unsupervised ones the segmentation is done with use of unlabeled data. Semi-supervised methods stand somewhere in between the aforementioned methods. In which, small amount of training data is labeled correctly and the rest is either unlabeled or falsely labeled.

Labeling the data requires an expert's knowledge and also is time consuming. Therefore, it is not feasible to utilize supervised methods for large amount of data. Semi-supervised algorithms provide an alternative solution to supervised methods from the computational cost point of view. They also give more accuracy compared to unsupervised methods [13].

Unsupervised methods usually use clustering procedure and do not depend on labeled training set. K-means and Fuzzy C-means are the two commonly used clustering algorithms [13], [14]. The purpose of clustering is to determine decision boundaries using unlabeled training data [15].
SECTION IV.
Segmentation Methods

Although a wide variety of segmentation methods have been presented, none of them will work perfectly for all imaging applications. Moreover, segmentation techniques are often optimized for specific imaging modality such as MRI or CT. Also, selection of the best suitable segmentation method for an intended application is a difficult task and is dependent to goal of segmentation task. In most cases a combination of different segmentation techniques is necessary. In this paper, we explain common segmentation methods used in recent medical image processing. After describing the procedure of each method, the advantages and drawbacks of that method are discussed. The methods can be classified into four classes: 1) Threshold based methods 2) Region based methods 3) Clustering and classification methods 4) Model-based methods
A. Threshold Based Methods

Thresholding is a simple, fast, and effective segmentation method. In this method, the intensities of objects in the image are compared to one (global threshold) or more thresholds (local thresholds).

When an image scan has a bimodal histogram (most common case when no pathology exists, i.e. normal body tissues), objects can be separated from the background with single threshold, also known as global threshold. However, when the image scan contains abnormality or when we are interested in separating different tissue types simultaneously, local thresholding is utilized, either by using several thresholds or by applying multiple thresholding techniques. The threshold value is obtained from the histogram using the valleys and peaks of the image's histogram.

Since thresholding only deals with the intensity of the pixels, not any other relationship between pixels, such as location, it does not consider the spatial information of images. This leads to sensitivity to noise and intensity inhomogeneities of the image. The mentioned problems cause some falsely brighter regions in the image and fundamental differences in the histogram which leads to more complex segmentation [16]–[17][18].
1) Global Thresholding

In this method the pixel intensities of image are compared to a single threshold value and by setting all the pixels above the threshold to one and below it to zero the image is segmented. This creates a binary image. This can be formulated as in (1).
g(x,y)={10f(x,y)≥Totherwise(1)
View SourceRight-click on figure for MathML and additional features. Where f(x, y) is the pixel intensity at (x, y) location, T is the selected threshold, and g(x, y) is the resulted binary image.

The selection of optimum threshold value is of great importance. One of the primary and useful methods in finding the optimal threshold value automatically is Otsu's thresholding method [18]. This method, first proposed in 1978, is based on discriminant analysis. The method assumes that the image has bimodal histogram [19], [20]. This method uses a grey-level histogram of input image and finds the optimal threshold value such that the overlap between the two class (object and background) is minimized.
2) Local Thresholding

To change the Local (adaptive) thresholding uses local threshold values instead of using a single threshold value for the entire image. In this method, intensity histogram is used to estimate the local threshold values. The estimation of threshold values is generally based on prior knowledge of image. Local threshold values can be estimated by using statistical properties such as mean intensity values used by Shanthi and Kumar [21].

Thresholding is the best choice, when an image contains objects with homogeneous region, or the contrast between the objects and the background is high. The selection of threshold becomes difficult when the image contrast is low or in the presence of noise. In general, threshold-based segmentation methods are used as the first step in a segmentation process, since they are unable to segment most medical images desirably [21]. Boegel et.al.[22] used gradient-based thresholding approach for blood vessel segmentation. In their approach, first, the parameters of a global thresholding algorithm is estimated using an iterative process. Then, a locally adaptive version of the approach is applied using the estimated parameters. The method showed promising results compared to common thresholding methods.
B. Region Based Methods

A group of connected pixels with similar properties form a region which may correspond to an object or part of an object. Region based methods form regions by merging neighborhood pixels. Computation of a region is based on predefined similarity criterion between pixels [23]. Region based methods divide the image into regions where two different regions have no overlap, and the union of the regions makes the original image. In fact, this method is the constrained version of the thresholding method in which the pixels within a region must satisfy the similarity criterion. The region growing segmentation algorithm is one of the most common region based methods used for brain tumor segmentation application.
1) Region Growing

Region growing (RG) is among the simplest interactive region based segmentation methods. At least one seed is required to initialize the algorithm. The seed is located at the structure of interest to be segmented. Neighbors of the seed are compared with the seed by means of similarity criterion one by one, and if any of them satisfies the condition then that neighbor is also added to the region. The process iterates until no more pixel can be added to the region. The similarity criteria can be intensity, or any other features in the image. Also automatic seed finding procedures, such as clustering, may be used to find the initial seeds.

While RG generates connected regions with simplicity, it is sensitive to the seed initialization. Some different methods have been proposed to reduce this sensitivity. Sajadi and Sabzpoushan [24] utilized cellular automata for the initial seed selection with conjunction of traditional RG and applied it for retinal blood vessels extraction. Also RG leaks to nearby tissues when the boundary information is not good enough and the image has poor contrast. RG performs well at homogenous and high contrast regions. Another disadvantage of the RG is that it is sensitive to noise and the segmentation produces disconnected areas in the presence of noise. However, RG is still used for many radiology applications such as lung, bone, and homogenous (isolated) tumor segmentation. It has also been widely used in extracting the potential lesions in mammograms [25].
C. Clustering and Classification Methods

Classification is a segmentation technique, in which training data is used to find patterns in the image. In classification methods, either supervised or unsupervised, classifier is used to cluster pixels in the feature space. Clustering means grouping data into classes such that high intra-class (in the same class) similarity and low inter-class (between different classes) similarity exists. The similarity is determined in terms of appropriate distance measure such as Euclidean distance measure. Each cluster is represented by its centroid or mean and variance. There are many clustering methods proposed in the literature. Basic unsupervised clustering methods are: Fuzzy C-means (FCM), K-means, and Markov Random Fields (MRF). The supervised clustering methods include artificial neural network (ANN), and Bayes methods. In this section FCM, MRF and ANN clustering techniques are presented and analyzed.
1) Fuzzy c-Means (fcm)

This Clustering method is an unsupervised algorithm which divides data into two or more clusters. This algorithm assigns membership to each data point corresponding to each cluster center. This membership is based on the distance between the cluster and the data point. The more near the data is to the cluster's center, the more possible its membership is towards that cluster center. Several FCM applications have been presented for MRI segmentation of different body parts [26], [27]. Since FCM is an iterative method, it is very time consuming. To overcome this problem, some solutions such as Bias corrected FCM (BCFCM) clustering algorithms are proposed. BCFCM segments brain images very quickly and provides good quality, which makes it excellent to support virtual brain endoscopy for brain tumor segmentation [28].
2) Markov Random Fields (MRF)

One of the unsupervised clustering algorithms, which integrate spatial information into clustering procedures, is MRF. In many applications, this reduces both the problem of clusters overlapping and noise effect on clustering results [29]. MRF is capable of handling complex dependencies among data instances, providing a high accuracy on segmentation tasks [30]. In [31] hidden MRFs are used to segment cone beam CT images of tumors.
3) Artificial Neural Networks (ANN)

This algorithm is a supervised clustering method. Mathematical operations are applied to the input nodes (features) of an ANN classifier, and the result is carried out at output nodes. To train ANN the values of parameters (involved in the mathematical operations) must be determined such that the error in the predictions (made by the output nodes) is minimized. ANN approaches are non-parametric techniques, since no parametric distribution (such as Gaussian) is assumed for the data. The use of hidden layers in ANN allows the modeling of non-linear dependencies in the input data. Although ANN training phase is complex, the ability to model non-trivial distributions gathers practical advantages. This is an important property in applications such as heterogeneous tumor segmentations where the simple Gaussian assumption is not appropriate [32], [33].

Many different approaches have been investigated in the classification and clustering algorithms segmentation in both feature and classifier selection. For example Dheeba et.al[35] followed a new classification approach for detection of breast abnormalities in digital mammograms using Particle Swarm Optimized Wavelet Neural Network (PSOWNN). The proposed abnormality detection algorithm is based on extracting Laws Texture Energy Measures from the mammograms and classifying the suspicious regions by applying a PSOWNN pattern classifier. ROC curve is then used to evaluate the performance of the method. Hassanien et.al [35] introduce a hybrid approach that combines the advantages of fuzzy sets, ant-based clustering and multilayer perceptron neural networks (MLPNN) classifier, in conjunction with statistical-based feature extraction technique. The authors applied the method for segmenting the breast cancer lesions from MR images.
D. Deformable Model Methods

The segmentation of 3D image data is a challenging task that has been mainly approached by model-based segmentation techniques as parametric/geometric deformable models. Deformable models have the ability to segment images of anatomic structures by building a connected and continuous model which takes into account a priori knowledge about the location, size, orientation, and shape of these structures.

Deformable models are capable of adjusting with significant variability of biological structures over time and across different individuals [36]. Existing deformable models can be divided into two classes: parametric and geometric. The following sections explain the parametric and geometric deformable models.
1) Parametric Deformable Models

Parametric deformable models (PDM) also known as snakes [37] are curves or surfaces defined within the image domain that move under the influence of internal and external forces. In fact, the PDM contour is a parametric curve (Esnake) defined as the sum of three energy terms shown in (2).

Where Econstraint, is the energy term considering constraint on the curve. Eextrenal, and Emtema1 are the energy terms corresponding to external and internal forces respectively.
Esnake=Einternal+Eexternal+Econstraint(2)
View SourceRight-click on figure for MathML and additional features.

The objective is to define the forces in a clever manner such that the final position of the contour will have a minimum energy. Therefore the problem boils down to an energy minimization problem.

Initialization plays an important role in snakes. In applications where the boundaries of objects are so close to each other, the initial position of the model should be placed close enough to the desired boundary to prevent converging to wrong boundaries.
2) Geometric Deformable Models

In PDMs (snakes, b- snakes, etc.) it is challenging to change the topology of the curve as it evolves. If the shape changes dramatically, curve re-parameterization may also be required. An alternative solution is to use geometric deformable models or Level sets (LS) [38].

LS evolve to fit and track objects of interest by modifying the underlying embedding function instead of modifying curve function as in snakes. These methods improve the initialization of parametric active contours and provide symmetrical placement of the initial contour with respect to the boundaries of interest. Although, in practice it is difficult to achieve good segmentation results because in many applications, objects of interest do not have regular shapes [39].
SECTION V.
Segmentation Evaluation

The validation of segmentation results is an important issue in medical image analysis since it has a direct impact on surgical decisions. There are two ways of evaluating the accuracy of segmentation results: qualitatively and quantitatively.

Qualitative evaluation is done by comparing the results to ground truth and giving some rating schemes.

Dice similarity Coefficient (DSC) is the most commonly used quantitative standard measure in many medical segmentation applications, where the overlap between the ground truth and the segmentation results is calculated. DSC is a number ranging from 0 to 1, with 0 indicating no overlap and 1 indicating complete overlap [40]. The drawback of DSC is that it is unsuitable for comparing segmentation accuracy on objects that differ in size [41]. In fact, DSC is not enough for evaluating the segmentation results [42]. For example, in Fig. 1, both of the segmentation results have similar DSC, but as it is clear, the result of segmentation in Fig. 1(b) is not acceptable due to low sensitivity and specificity. Therefore, other measurements such as sensitivity and specificity are calculated along with DSC to give better evaluation of the segmentation results. Sensitivity measures the overlap, but for a poor segmentation much bigger than the ground truth it can be equal to 1. Specificity is therefore the necessary and complementary part of the sensitivity, but it can be equal to 1 for a very poor segmentation that does not detect the object of interest at all.
Figure 1
Fig. 1

Schematic diagrams of sensitivity and specificity metrics with color-coded condition—test outcome pairs: true positive (tp) (green area), true negative (tn) (white area), false positive (fp) (yellow area), and false negative (fn) (blue area). (a) Sensitivity = 94.69% specificity = 94.19%. (b) Sensitivity = 72.99% specificity = 78.16%. VGT = ground truth segmentation, vtest = performed segmentation (brown area) [42]

DSC is defined in (3), while sensitivity and specificity are defined in (4) and (5). In the equations, VGT is the reference standard segmentation (ground truth), Vseg is the segmentation performed by using any of the methods, |⋅| is the size operator, and ⋂ determines the overlapping area between the VGT and the segmentation. TP indicates true positives (intersection between segmentation and ground truth), FP corresponds to false positives (segmented parts not overlapping the ground truth), FN shows the false negatives (missed parts of the ground truth), and TN stands for true negatives (part of the image beyond the union of segmentation and ground truth).
DSC=2|VGT∩Vseg||VGT|+|Vseg|Sensitivity=TPTP+FNSpecificity=TNTN+FP(3)(4)(5)
View SourceRight-click on figure for MathML and additional features.

There are other complementary evaluation metrics such as Hausdorff distance measures which are used for measuring boundary mismatches in the literature [43].

The evaluation of segmentation algorithms always depends on the specific task. For instance, in the tumor segmentation application the requirements are different when the next procedure is to do surgery compared to the volumetric chemotherapy follow-up assessment.
SECTION VI.
Conclusion

This paper explains a number of current image processing methods which are widely used in medical image segmentation. The appropriate segmentation algorithm should be chosen based on the application of the segmentation. When regions of interest can be distinguished from the background by their intensity, threshold-based or region growing techniques have been employed. On the other hand, when objects can be identified by their shapes, model-based techniques are applied for the segmentation. As recently reported in the literature, several general conclusions can be drawn. First, for the sake of reproducibility, it is important to direct segmentation towards a fully automated method by using human intelligence and prior knowledge about the structure of interest along with the algorithm to improve the segmentation results. Second, the use of some pre- or postprocessing methods has demonstrated improved segmentation results. Third, since to the best of our knowledge there is not a standardized database available for many medical applications, the comparison between different algorithms is difficult. Even if such a unified database exists for some medical challenges (like BRATS database for brain tumor segmentation [44]) only a few of the current algorithms were applied to such databases. Finally, although current segmentation algorithms show promising results, the automatic segmentation methods still need more robust procedures to gain wide acceptance among the clinicians for every day clinic practice.

-----------
Evaluation metrics
For detection use this:
https://pdfs.semanticscholar.org/d051/ba0a904e4b4c45a2af145aa29b8490bbbc5c.pdf
For recognition this might be a good idea to start with:
http://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
