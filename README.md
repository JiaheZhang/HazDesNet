# HazDesNet
An End-to-End Network for Haze Density Prediction [1]

## Abstract

Vision-based intelligent systems such as driver assistance systems and transportation systems should take into account weather conditions. The presence of haze in images can be a critical threat to driving scenarios. Haze density measures the visibility and usability of hazy images captured in real-world conditions. The prediction of haze density can be valuable in various vision-based intelligent systems, especially in those systems deployed in outdoor environments. Haze density prediction is a challenging task since the haze and many scene contents have a lot in common in appearance. Existing methods generally utilize different priors and design complex handcrafted features to predict the visibility or haze density of the image. In this article, we propose a novel end-to-end convolutional neural network (CNN) based method to predict haze density, named as HazDesNet. Our HazDesNet takes a hazy image as input and predicts a pixel-level haze density map. The density map is then refined and smoothed, and the average of the refined map is calculated as the global haze density of the image. To verify the performance of HazDesNet, a subjective human study is performed to build a Human Perceptual Haze Density (HPHD) database, which includes 500 real-world hazy images and 100 synthetic hazy images, and the corresponding human-rated perceptual haze density scores. Experimental results show that our method achieves the best haze density prediction performance on our built HPHD database and existing databases. Besides the global quantitative results, our HazDesNet is capable of predicting a continuous, stable, fine, and high-resolution haze density map. We will make the database and code publicly available at https://github.com/JiaheZhang/HazDesNet.

The paper has been accepted by T-ITS and can be found at [this](https://ieeexplore.ieee.org/document/9237140/)

## Datasets

The Human Perceptual Haze Density (HPHD) database includes a real-world hazy image (RHI) subset and a synthetic hazy image (SHI) subset. The database can be downloaded at [this](https://github.com/JiaheZhang/HazDesNet/releases/tag/0.0.2). The details of the database are described in the paper. Besides, we also provide a copy of the testing hazy image and MOSs of the [LIVE Image Defogging Database](https://github.com/JiaheZhang/HazDesNet/releases/download/0.0.1) [2] to facilitate the research. If you want to use the whole database, please visit the website in [2] and follow their rules.

## Uages

### 1. Prepare

Download the database and install the Python enviroments according to the requirements.txt. We use Keras package and Tensorflow backend.

### 2. Prediction

Use predict.py to predict the haze density score and map for hazy images.

```python
python predict.py
```

### 3. Evaluation
Download the databases and put them in the ./dataset dir.

```python
python eval.py
```

## Results

Comparison between FADE [3] and HazDesNet.

![Comparison](images/comparison/cmp.png?raw=true)


## References

[1] J. Zhang et al., "HazDesNet: An End-to-End Network for Haze Density Prediction," in IEEE Transactions on Intelligent Transportation Systems, doi: 10.1109/TITS.2020.3030673.

[2] L. K. Choi, J. You, and A. C. Bovik, "LIVE Image Defogging Database," Online: http://live.ece.utexas.edu/research/fog/fade_defade.html, 2015.

[3] L. K. Choi, J. You, and A. C. Bovik, "Referenceless Prediction of Perceptual Fog Density and Perceptual Image Defogging," IEEE Transactions on Image Processing, vol. 24, no. 11, pp. 3888-3901, Nov. 2015.