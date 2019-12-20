# ComputerVisiononDeepLearning
- This repo contains the basis of the Computer Vision applied with Deep Learning Models especially with Convolutional Deep Learning (CNN)

## Contents
  - 1. **Computer Vision Overview**
  - 2. **Face Detection with Vialo-Jones Algorithm**
    - 2.1 **Vialo-Jones Algorithm Overview**
    - 2.2 **Haar-like Features**
    - 2.3 **Integral Images**
    - 2.4 **Training Classifiers**
    - 2.5 **Cascaeding**
  - 3. **Object Detection**
    - 3.1 **Single Shot Multibox Detector (SSD)**
    - 3.2 **Preceding Object Positions**
    - 3.2 **Scale Problem**
  - 4. **Generative Adversarial Networks (GANs) Intuition**
     - 4.1 **GANs Working Process**
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
- Basic Haar-like features are **edge,linear,four regtangle haar-like features.
- In face detection,in order to detect the object as a face we need to determine human's nose,eyes,lip etc.
- For instance when we convert to RGB frontal face image into a gray scale image.We can see that some parts are bright and some of them is black.For lip of a human we can say that it has a **linear haar-like feature**.
- We need to make sure that it is a correct haar-feature or not.Because of this,generally we determine a **threshold** value.If the value slides up it then we can say that it is a haar feature that work for evaluation but,of course it is not enough.

- **2.3 Integral Images**

 
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

