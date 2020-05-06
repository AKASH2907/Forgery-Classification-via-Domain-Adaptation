# Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation

[![WACVW](https://img.shields.io/badge/WACVW-CVF-b31b1b.svg)](http://openaccess.thecvf.com/content_WACVW_2020/html/w4/Kumar_Syn2Real_Forgery_Classification_via_Unsupervised_Domain_Adaptation_WACVW_2020_paper.html)

This repository provides the official Python implementation of Syn2Real: Forgery Classification via Unsupervised Domain Adaptation. [(Link)](http://openaccess.thecvf.com/content_WACVW_2020/html/w4/Kumar_Syn2Real_Forgery_Classification_via_Unsupervised_Domain_Adaptation_WACVW_2020_paper.html) In this work, using Domain Adversarial Neural Network (DANN) and Deep Domain Confusion (DDC) Domain Adaptation networks, we adapt to the features from a synthetically generated dataset onto a realistic dataset. Our main focus is generalizability across forgery detection in unsupervised conditions, keeping in view to improve the accuracy scores too. 

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

## Dependencies

Tested on Python 3.6.x and Keras 2.3.0 with TF backend version 1.14.0.
* Numpy (1.16.4)
* OpenCV (4.1.0)
* Pandas (0.25.3)
* Scikit-learn (0.22.1)
* PyTorch (1.2.0)

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

1) **Semantic Inpainting:** We used 80 sub-categories of COCO dataset to create a forged dataset. We take mask of each category and cut them out. Then, we fill those region via Deep Semantic Inpainting. In this way, the image looks natural as well as it make the network focus on edge discrepancies around the forged region. The figure below presents an overview of semantic inpainting dataset generation approach:

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569484-354df480-5a7b-11ea-8f9e-eda5b54c6253.png">
</p>

* To generate inpainted images, have a look in this repository:-> [Edge-Connect](https://github.com/knazeri/edge-connect#2-testing)

2) **Copy-Move Forgery:** Images alongwith their segmentation is mask selected. We compare the mask of all the areas. Keeping a minimum threshold, we select the mask with the largest area. We apply a image matting so that pasted region could easily blend in. Overnight 60,000 images can be generated.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569534-4d257880-5a7b-11ea-8636-3495d521d478.png">
</p>

* For CMF data generation, please look into my other repository:-> [Synthetic data Generation](https://github.com/AKASH2907/synthetic_dataset_generation)

### 2. Domain Adaptation 

1) DANN: It has two separate heads: Source classifier Head and Domain classifier head.
  * Source Head: Feature parameters(ϴ<sub>f</sub>) and label classifier parameter optimized to reduce classification loss.
  * Domain Head: Feature parameters maximizes domain loss to make distributions similar.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569727-b0170f80-5a7b-11ea-9d33-7ea3c6467d24.png">
</p>

DANN Loss function:

<p align="center">
  <img height="100" src="https://user-images.githubusercontent.com/22872200/75612024-d2755f80-5b45-11ea-9c96-f68e512c6cbc.png">
</p>

2) DDC: Minimizes the distance between source and target distribution via Maximum Mean Discrepancy (MMD) loss. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569759-c1f8b280-5a7b-11ea-8740-a4b1e0b75de5.png">
</p>

DDC Loss Function:

<p align="center">
  <img height="35" src="https://user-images.githubusercontent.com/22872200/75612038-f5a00f00-5b45-11ea-9809-7052ae5a938d.png">
</p>

## Dataset Information

| Image Properties  | **CASIA** | **CoMoFoD** |
| :------------:| :------------:| :------------:|
| Resolution | 240x160 - 900x600 | 512x512 |
| # pristeine/ tampered | 1701/3274 | 200/200 |
| Image Format  | JPG, TIFF | PNG |
| Post-image processing | Translation, Rotation, Scaling, Affine Transformation | Translation, Rotation, Scaling, Affine Transformation




## Experiments

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75570108-77c40100-5a7c-11ea-8ddf-e03f00fb27a7.png">
</p>

***COCO->CASIA:***
* Train/Test Distribution: 4000/1000 images
* More images -> Able to optimize huge number of parameters
* DANN+VGG-7 outperforms others

***COCO->CoMoFoD:***
* Train/Test Distribution: 200/200 images
* Less images -> Can't optimize huge number of parameters
* DDC+AlexNet outperforms others

## Results

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75569630-7cd48080-5a7b-11ea-9403-b95c5791f0af.png">
</p>

* Compare to BusterNet where they have used **1 lakh** images for supervised training, we used **40k** images to achieve better accuracy.
* Our approach using **Domain Adaptation** improves the previously reported baseline.

## References
[1] Ganin, Yaroslav et al. “Domain-Adversarial Training of Neural Networks.” J. Mach. Learn. Res. 17 (2015). [Link](https://arxiv.org/abs/1505.07818) </br>
[2] Tzeng, Eric et al. “Deep Domain Confusion: Maximizing for Domain Invariance.” ArXiv abs/1412.3474 (2014). [Link](https://arxiv.org/abs/1412.3474) </br>
[3] Nazeri, Kamyar et al. “EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning.” ArXiv abs/1901.00212 (2019). [Link](https://arxiv.org/abs/1901.00212)

## Citation

If you use this repository, please use this bibtex to cite the paper:
 ```
@InProceedings{Kumar_2020_WACV,
author = {Kumar, Akash and Bhavsar, Arnav and Verma, Rajesh},
title = {Syn2Real: Forgery Classification via Unsupervised Domain Adaptation},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV) Workshops},
month = {March},
year = {2020}
} 
```

## Future Work
- [ ] https://github.com/wuhuikai/GP-GAN -> Image Blending using GANs in high resolution images.
- [ ] Improve the precision score keeping the high recall.
