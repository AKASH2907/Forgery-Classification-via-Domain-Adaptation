# Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation
Classifying Forged vs Authentic via Domain Adaptation Algorithm (DANN)

In this work, we've used Unsupervised Domain Adapatation Algorithm, Domain Adversarial Neural Network (DANN) to improve the accuracy of classification of copy-move forged images. We created a dataset of 15,000 of forged images from COCO dataset. 

In domain adaptation, we adapt the target domain feature space to source domain feature space, such that the features remains discriminative amongst classes but the domains becomes invariant. In our case, source domain is COCO forged dataset and target domain is CASIA V2 dataset. Our sorce domain contains of 15,000 images, half of which are authentic and the other half is forged. In target domain, CASIA contains 1300 authentic and 3300 copy-move forged images.

We can't apply direct transfer learning in this case. Mainly, because of two reasons:
* The number of images are less and the number of parameters needed are huge. It simply overfits the dataset and the test time perfromance is very poor.
* Pre-trained archirectures are trained on ImageNet dataset in which there is provision for forged images as such. So, it may not perform well in our case.

## Getting Started

* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
 ```
* [data_prepare.py](https://github.com/AKASH2907/Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation/blob/master/data_prepare.py) - 

* [dann_keras.py](https://github.com/AKASH2907/Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation/blob/master/dann_keras.py) - 

* [models.py](https://github.com/AKASH2907/Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation/blob/master/models.py) - 

## Dataset Creation

We used 80 sub-categories of COCO dataset to create a forged dataset. We take mask of each category and cut them out. Then, we fill those region via Deep Semantic Inpainting. In this way, the image looks natural as well as it fullfills our pupose too. The figure below presents an overview for dataset generation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/66417249-383d3f80-ea1d-11e9-8726-727e1720f768.png">
</p>

## Architecture

We used Domain Adversarial NN for unsupervised Domain Adaptation algorithm. The architecture we used in depicted in figure below: 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/66037369-162f5300-e52d-11e9-9dbe-00c93d1b332e.png">
</p>


## Experiments

## Results

## References


## Future Work
1) https://github.com/wuhuikai/GP-GAN -> Image Blending using GANs in high resolution images
