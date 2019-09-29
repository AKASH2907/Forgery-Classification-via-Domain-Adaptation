# Copy-Move-Forgery-Classification-via-Unsupervised-Domain-Adaptation
Classifying Forged vs Authentic via Domain Adaptation Algorithm (DANN)

In this work, we've used Unsupervised Domain Adapatation Algorithm, Domain Adversarial Neural Network (DANN) to improve the accuracy of classification of copy-move forged images. We created a dataset of 15,000 of forged images from COCO dataset. 

In domain adaptation, we adapt the target domain feature space to source domain feature space, such that the features remains discriminative amongst classes but the domains becomes invariant. In our case, source domain is COCO forged dataset and target domain is CASIA V2 dataset. Our sorce domain contains of 15,000 images, half of which are authentic and the other half is forged. In target domain, CASIA contains 1300 authentic and 3300 copy-move forged images. 
