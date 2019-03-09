# CNN_basics
Simple repository for beginners who want to explore the possibilities of CNNs.

This repository contains an example of how to deal with images in python, how to format them for CNNs and how to train you network to recognize the images. 

In this example I have gatherd a small set of images. The one of the sets contain an assortment of pictures of rubber ducks and the other one contais pictures of stop signs. The reasoning for these two categories is that I'm hoping that the network can notice the straigth lines of the stop signs and the round edges of the ducks to easily differentiate between these two objects. 

The dataset is extremely small, and I have cropped the pictures to only be 100x100, partly to reduce dimensionality and partly so that this network can be run easily on computers wihtout an NVIDIA gpu. 

I have also provided the model for you to try out if you do not want to train your own.

The dataset I used is scraped from Google image search by me. I do not own the images so I wont post them here, but if you wish to use my dataset feel free to contact me and I'll see what I can do.

Below I have attached results of the network:

# Training results
![alt text](https://i.imgur.com/h5cu3O6.png)

![alt text](https://i.imgur.com/NTNQO9s.png)

![alt text](https://i.imgur.com/o0y9YCe.png)

