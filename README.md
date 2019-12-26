# Computer Vision on Deep Learning
- This repo contains the basis of the Computer Vision applied with Deep Learning Models especially with Convolutional Deep Learning (CNN)

## Contents
  - 1. **Computer Vision Overview**
  - 2. **Face Detection with Vialo-Jones Algorithm**
    - 2.1 **Vialo-Jones Algorithm Overview**
    - 2.2 **Haar-like Features**
    - 2.3 **Integral Images**
    - 2.4 **Training Classifiers**
  - 3. **Object Detection**
    - 3.1 **Major Problem of Object Detection with CNN**
    - 3.2 **Single Shot Multibox Detector (SSD)**
    - 3.3 **R-CNN Object Detection**
    - 3.4 **Fast R-CNN**
    - 3.5 **YOLO**
  - 4. **Generative Adversarial Networks (GANs) Intuition**
  - 5. **Convolutional Neural Network for Computer Vision**
  
## Computer Vision Overview
- Computer Vision, often abbreviated as CV, is defined as a field of study that seeks to develop techniques to help computers see and understand the content of digital images such as photographs and videos.
- It is a multidisciplinary field that could broadly be called a subfield of artificial intelligence and machine learning, which may involve the use of specialized methods and make use of general learning algorithms.
- A given computer vision system may require image processing to be applied to raw input, e.g. pre-processing images.For instance to detect the face,Vialo-Jones algorithm first convert image to gray scale to reduce pixel process.

## 2.Face Detection with Vialo-Jones Algorithm
- **2.1 Vialo-Jones Algorithm Overview**
- Vialo and Jones are two scientist that they had developed one of the most powerfull computer vision face detection algorithm in 2001 and it still has this popularity.
- The algorithm can be used for both images and video frames.However,it shows it best efficiency in frontal face images.
- The algorihm is consist of two phases **training**,**detection** respectivey.In **training phase** generally,a strongest classifier is tried to be obtained from samples which consist of relevant and irrelevant images.
- Algorithm uses **Haar-like features** to obtain an **integral image**

- **2.2 Haar-Like Features**
- Haar is Hungarian mathmetician and he has found the features can be used to detect important features in an image.
- Basic Haar-like features are **edge,linear,four regtangle haar-like features**.
- In face detection,in order to detect the object as a face we need to determine human's nose,eyes,lip etc.
- For instance when we convert to RGB frontal face image into a gray scale image.We can see that some parts are bright and some of them is black.For lip of a human we can say that it has a **linear haar-like feature**.
- We need to make sure that it is a correct haar-feature or not.Because of this,generally we determine a **threshold** value.If the value slides up it then we can say that it is a haar feature that work for evaluation but,of course it is not enough.

- **2.3 Integral Images**
- This is a trick to get the haar-feature as quickly as possible.
- Before step into integral images I want to talk about the basic process of haar-feature obtaining.
- Let's say we have edge haar-like feature on nose of human.Actually this feature is full of pixels.
- Those pixels have both brighter and darker sides and we think that we work on a gray scale image in accordiance with a Viola-Jones algorithm.
- First we give values to those pixels as normalized values in range 0 to 1.Then we divide the feature as white and black side.-
- The next step is to calculate an average value for both side.Let's say we have M pixels on both sides.
- Then, we get the subtract two value from each other.If this value is greater than the threashold value we determine,then we can say that it is an **haar-like feature**.
- This is a long and very expensive process since,haar-likle features can be expanded and shrunk.This means in such a small feature we can have tousands of pixels inside.
- To make the process a bit more simple and we use integral image.**Integral image** has the same size with original image and every pixel value is the sum of all pixels of its above and left.Then subtract some pixels blocks from each other.That means via small size of pixels we use,we can determine whether it is an haar-like feature or not.

- **2.4 Training Classifiers && Cascading**
- To determine the detection as the first part of the algorithm we need to train our features.To get the most important features we need
others that complement each other. **F(X)=a1f(x1)+a2f(x2)+...+anf(xn)** ,where F(X) is called **strongest training classifier** and the others called **weak training classifiers**.
- While evaluating whether it is a part of the human face or not,we need many samples.For instance we detected that it is a nose feature but it is not enough.We need to check many more human faces for training.Plus,we need also False negative and false negative features to be used as weak classifier samples to make the prediction more robust and price as much as possible.

## 3.Object Detection
- **3.1 Major Problem of Object Detection with CNN**
- The major reason why you cannot proceed with this problem by building a standard convolutional network followed by a fully connected layer is that, the length of the output layer is variable
-  The major reason why you cannot proceed with this problem by building a standard convolutional network followed by a fully connected layer is that, the length of the output layer is variable — not constant, this is because the number of occurrences of the objects of interest is not fixed. A naive approach to solve this problem would be to take different regions of interest from the image, and use a CNN to classify the presence of the object within that region. The problem with this approach is that the objects of interest might have different spatial locations within the image and different aspect ratios. Hence, you would have to select a huge number of regions and this could computationally blow up. Therefore, algorithms like R-CNN, YOLO etc have been developed to find these occurrences and find them fast.
- **3.2 Single Shot Multibox Detector**
- SSD’s architecture builds on the venerable **VGG-16 architecture**, but discards the fully connected layers.VGG is a innovative
object detection model.
- VGG has strong performance on high quality image tasks.
- Instead of the original VGG fully connected layers, a set of auxiliary convolutional layers were added, thus enabling to extract features at multiple scales and progressively decrease the size of the input to each subsequent layer.
- To have more accurate detection, different layers of feature maps are also going through a small 3×3 convolution for object detection 
- Say for example, at **Conv4_3**, it is of size 38×38×512. 3×3 conv is applied. And there are 4 bounding boxes and each bounding box will have (classes + 4) outputs. Thus, at Conv4_3, the output is 38×38×4×(c+4). Suppose there are 20 object classes plus one background class, the output is 38×38×4×(21+4) = 144,400. In terms of number of bounding boxes, there are 38×38×4 = 5776 bounding boxes.
Similarly for other conv layers:
- Conv7: 19×19×6 = 2166 boxes (6 boxes for each location)
- Conv8_2: 10×10×6 = 600 boxes (6 boxes for each location)
- Conv9_2: 5×5×6 = 150 boxes (6 boxes for each location)
- Conv10_2: 3×3×4 = 36 boxes (4 boxes for each location)
- Conv11_2: 1×1×4 = 4 boxes (4 boxes for each location)
- If we sum them up, we got 5776 + 2166 + 600 + 150 + 36 +4 = 8732 boxes in total. If we remember YOLO, there are 7×7 locations at the end with 2 bounding boxes for each location. YOLO only got 7×7×2 = 98 boxes. Hence, SSD has 8732 bounding boxes which is more than that of **YOLO**.
- **3.3 RCNN**
- Proposed a method where we use selective search to extract just 2000 regions from the image and he called them **region proposals**.
- Therefore, now, instead of trying to classify a huge number of regions, you can just work with 2000 regions. These 2000 region. proposals are generated using the selective search algorithm which is written below.
- **Selective search** applied by these steps:
1. Generate initial sub-segmentation, we generate many candidate regions
2. Use greedy algorithm to recursively combine similar regions into larger ones 
3. Use the generated regions to produce the final candidate region proposals 
- These 2000 candidate region proposals are warped into a square and fed into a convolutional neural network that produces a 4096-dimensional feature vector as output.
- The extracted features are fed into an **SVM** to classify the presence of the object within that candidate region proposal.
- **Problems with R-CNN**
- It still takes a huge amount of time to train the network as you would have to classify 2000 region proposals per image.
- It cannot be implemented real time as it takes around 47 seconds for each test image.
- The selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.

- **3.4 F-RCNN**
- Similar to Fast R-CNN, the image is provided as an input to a convolutional network which provides a convolutional feature map. Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals. The predicted region proposals are then reshaped using a RoI pooling layer which is then used to classify the image within the proposed region and predict the offset values for the bounding boxes.

-All of the previous object detection algorithms use regions to localize the object within the image. The network does not look at the complete image. Instead, parts of the image which have high probabilities of containing the object. YOLO or You Only Look Once is an object detection algorithm much different from the region based algorithms seen above. In YOLO a single convolutional network predicts the bounding boxes and the class probabilities for these boxes.

- **3.5 YOLO (You Only Look Once)**
- YOLO is orders of magnitude faster(45 frames per second) than other object detection algorithms. The limitation of YOLO algorithm is that it struggles with small objects within the image, for example it might have difficulties in detecting a flock of birds. This is due to the spatial constraints of the algorithm.

## **4.Generative Adversarial Networks (GANs) Intuition**
- It is a neural network that enquires the probability of x as p(x|y).
- It consists of two steps **Generation** and **Discriminator**.
- It generates objects by **Generation Phase** such as images,videos,3D animations and these objects are assesed by the **discriminator**
- Its usage fields are Speech Generation,Assisting Artists,Generating Images,Face Ageing.

## Announcement
- Overview of Deep Learning, **Dimension Reduction** , **Model Selection** , **XGBoot** topics will be under **Deep Learning Repo** 
- **Convolutional Neural Networks (CNN)** will be under **Artificial Intelligence Repo (AI)** 
- **Computer Vision** , **Self Autonomous Driving** with Tensorflow-Keras & Computer Vision & Deep Learning Repos will be also shared 
- **Kubernates** repo will be also shared 
- You can also check out Java Spring Framework series which will include **Spring Core,Spring MVC,Spring Boot** repo under
[Java Spring Framework Repo](https://github.com/ozanyldzgithuboffical/Spring)
- You can also check out Machine Learning series which will include **Machine Learning Basis,Prediction,Pre-Processing,Regression,Classification,Clustring & Reinforcement Learning** techniques.
[Machine Learning Repo](https://github.com/ozanyldzgithuboffical/OzanYldzML)
- You can also check out Natural Language Processing (NLP) series.
[Natural Language Processing (NLP)](https://github.com/ozanyldzgithuboffical/NLP-Natural-Language-Processing-)
- You can also check out Deep Learning Series.
[Deep Learning Repo](https://github.com/ozanyldzgithuboffical/DeepLearning)
- **Spring Microservices with Spring Cloud** repo will be also available later. 
- **Docker** repo will be also available later.

## About the Repo
- This repo is open-source and aims at giving an overview about the top-latest topics that will lead learning of the basis of deep learning and intelligent systems basis.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. Thanks.

**OZAN YILDIZ**
-Computer Engineer at HAVELSAN Ankara/Turkey 
**Linkedin**
[Ozan YILDIZ Linkedin](https://www.linkedin.com/in/ozan-yildiz-b8137a173/)

## License
[MIT](https://choosealicense.com/licenses/mit/)

