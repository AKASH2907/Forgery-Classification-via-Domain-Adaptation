# Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation

In this work, using Domain Adversarial Neural Network (DANN) and Deep Domain Confusion (DDC) Domain Adaptation networks, we adapt to the features from a synthetically generated dataset onto a realistic dataset. Our main focus is generalizability across forgery detection in unsupervised conditions, keeping in view to improve the accuracy scores too. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75251987-bef28d80-5801-11ea-9a15-7625e621368a.png">
</p>

The repository includes:
* Generating Copy-Move forgery snthetic data
* Training dataset preparation
* Training and testing code for DANN
* Training and testing code for DDC
* Base network models for feature extraction

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this paper (bibtex below). 

In domain adaptation, we adapt the target domain feature space to source domain feature space, such that the features remains discriminative amongst classes but the domains becomes invariant. In our case, source domain is COCO forged dataset and target domain is CASIA V2 and CoMoFoD dataset. Our sorce domain contains of 40,000 images, half of which are authentic and the other half is forged. In target domain, CASIA contains 1300 authentic and 3300 copy-move forged images and CoMoFoD has 200 authentic and 200 forged.

We can't apply direct transfer learning in this case. Mainly, because of two reasons:
* The number of images are less and the number of parameters needed are huge. It simply overfits the dataset and the test time perfromance is very poor.
* Pre-trained archirectures are trained on ImageNet dataset in which there is provision for forged images as such. So, it may not perform well in our case.

## Getting Started

* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
 ```
* [dataset_generation.py]() Generates Copy-Move forged dataset utilizing COCO dataset.

* [data_prepare.py](https://github.com/AKASH2907/Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation/blob/master/data_prepare.py) - Generate numpy arrays of training and testing datasets.

* [dann.py](https://github.com/AKASH2907/Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation/blob/master/dann_keras.py) - Train the DANN model using AlexNet or VGG-7 as the base feature extraction architecture.

* [ddc.py]() - Train and test the DDC model using AlexNet and VGG-7 feature extractors.

* [models.py](https://github.com/AKASH2907/Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation/blob/master/models.py) - AlexNet and VGG-7 base architecture models.

## Step by Step Domain Adaptation

### 1. Dataset Generation

We used 80 sub-categories of COCO dataset to create a forged dataset. We take mask of each category and cut them out. Then, we fill those region via Deep Semantic Inpainting. In this way, the image looks natural as well as it fullfills our pupose too. The figure below presents an overview for dataset generation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569484-354df480-5a7b-11ea-8f9e-eda5b54c6253.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569534-4d257880-5a7b-11ea-8636-3495d521d478.png">
</p>

### 2. Domain Adaptation 

We used Domain Adversarial NN for unsupervised Domain Adaptation algorithm. The architecture we used in depicted in figure below: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569727-b0170f80-5a7b-11ea-9d33-7ea3c6467d24.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569759-c1f8b280-5a7b-11ea-8740-a4b1e0b75de5.png">
</p>

## Experiments

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75570108-77c40100-5a7c-11ea-8ddf-e03f00fb27a7.png">
</p>

## Results

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569630-7cd48080-5a7b-11ea-9403-b95c5791f0af.png">
</p>


## References


## Future Work
- [ ] https://github.com/wuhuikai/GP-GAN -> Image Blending using GANs in high resolution images.
- [ ] Improve the precision score keeping the high recall.
