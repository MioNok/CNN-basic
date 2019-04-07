# CNNs
Added a new CNN with upgrades on almost every front. Better code in general, more images and a better model in every way. This model is based on a dataset from kaggle: https://www.kaggle.com/alexattia/the-simpsons-characters-dataset/kernels

I managed to get an accuracy of **89%** when evaluating on test set. The model is uploaded, and you can try it out on your own with some modifications to the duck model load file.

### Co-occurrence matrix of simpsons
![alt text](https://i.imgur.com/oE2mXv0.png)

Most of the errors were between bart and lisa, probably since their shapes/colors are similar.


## Old stuff
### Duck and stop sign dataset

_In this example I have gatherd a small set of images. The one of the sets contain an assortment of pictures of rubber ducks and the other one contais pictures of stop signs. The reasoning for these two categories is that I'm hoping that the network can notice the straigth lines of the stop signs and the round edges of the ducks to easily differentiate between these two objects._ 

_The dataset is extremely small, and I have cropped the pictures to only be 100x100, partly to reduce dimensionality and partly so that this network can be run easily on computers wihtout an NVIDIA gpu._ 

_I have also provided the model for you to try out if you do not want to train your own._

_The dataset I used is scraped from Google image search by me. I do not own the images so I wont post them here, but if you wish to use my dataset feel free to contact me and I'll see what I can do._


## Snapshot of the datasets

### Stop signs:
![alt text](https://i.imgur.com/FBtHM9g.png)

### Rubber Ducks:
![alt text](https://i.imgur.com/1tRRhDk.png)


## Training results
![alt text](https://i.imgur.com/h5cu3O6.png)

![alt text](https://i.imgur.com/NTNQO9s.png)

![alt text](https://i.imgur.com/o0y9YCe.png)





