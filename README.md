# Report

## Author
Andrii Kozar, student of Poznan University of Technology.

## Problem
<p>The problem's name is Semantic Image Segmentation. In an image classification task, the network assigns a label (or class) to each input image. However, suppose you want to know the shape of that object, which pixel belongs to which object, etc. In this case, you need to assign a class to each pixel of the image — this task is known as segmentation. A segmentation model returns much more detailed information about the image. Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging, just to name a few.</p>
<p>Once more, the goal of the semantic segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we’re predicting for every pixel in the image, this task is commonly referred to as dense prediction. The expected output in semantic segmentation are not just labels or bounding box parameters, but a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. Thus it is a pixel level image classification.</p>
<p>To solve Semantic Segmentation problem, we use neural networks.</p>

## Dataset
The dataset used in problem solving task is ADE20K. It is composed of more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. The dataset contains images with appropriate masks. Below you can see already resized image and its segmentation mask.
![exmpl](https://github.com/Hatter01/semantic_segmentation/blob/main/images/Dataset_exmp.png)

### Dataset stats used in project
The current version of the dataset contains:
<ul>
  <li>22,210 images (20,210 for training and 2,000 for validation).</li>
  <li>3,489 images without masks for testing (taken from separate unofficial release) </li>
  <li>150 categories (~classes)</li>
</ul>

### Download dataset
Running download_set.py script, you should be able to download ADE20K dataset. All scripts from this project should be in one folder to work properly. Then the dataset will be also downloaded to this folder.

## CNN architectures 
<p>In our experiments we are working with 2 cnn architectures: U-Net and FCNN. They both are implemented and trained from scratch.</p>

### FCNN
<p>CNN consists of a convolutional layer, a pooling layer, and a non-linear activation function. In most cases, CNN has a fully connected layer at the end in order to make class label predictions. But when it comes to semantic segmentation, we usually don’t require a fully connected layer at the end because our goal isn’t to predict the class label of the image. In semantic segmentation, our aim is to extract features before using them to separate the image into multiple segments. However, the issue with convolutional networks is that the size of the image is reduced as it passes through the network because of the max-pooling layers. To efficiently separate the image into multiple segments, we need to upsample it using an interpolation technique, which is achieved using deconvolutional layers. In general AI terminology, the convolutional network that is used to extract features is called an encoder. The encoder also downsamples the image, while the convolutional network that is used for upsampling is called a decoder.</p>

In this model we use pre-trained VGG-16 for feature extraction, adding on top some layers to concatenate and upsample the feature maps into pixel-level predictions. We also replace VGG dense layers by convolution layers and merge them. In short, you can see [FCNN structure](https://github.com/Hatter01/semantic_segmentation/blob/main/images/fcnn_str.png). The total number of parameters is 22,269,973.

### U-Net
The U-Net is a modification of a fully convolutional network (FCNN). It was introduced by Olaf Ronneberger et al. in 2015 for medical purposes—primarily to find tumors in the lungs and brain. The U-Net has a similar design of an encoder and a decoder. The former is used to extract features by downsampling, while the latter is used for upsampling the extracted features using the deconvolutional layers. The only difference between the FCNN and U-Net is that the FCNN uses the final extracted features to upsample, while U-Net uses something called a shortcut connection to do that. Here you can find detailed [U-Net structure](https://github.com/Hatter01/semantic_segmentation/blob/main/images/unet_str.png). It has total 34,523,095 parameters and all are trainable.

## Training process, hyperparameters
<p>GPUs and TPUs can radically reduce the time required to execute a single training step. Achieving peak performance requires an efficient input pipeline that delivers data for the next step before the current step has finished. The tf.data.Dataset API helps to build flexible and efficient input pipelines. After loading of ADE20K, we create training and validation tf.data.Dataset and parse images inside them. Then we shuffle training dataset. We use such parameters as batch_size=5 (in batch method) and buffer_size=1000 (when shuffle dataset).</p>
<p>When it comes to images, we resize them for both models to 128x128 since U-Net and FCNN take input sizes which need to be divisible with 32. After that we use data augmentation, or in our case we randomly flip images and their masks. In the end we normalize images by dividing by 255.</p>
<p>Now the dataset is ready and we can define our model and train it. When it comes to hyperparameters, we use the following:
<ul><li>Taken number of epochs is 10 (just to observe if the model is learning). Optimal number of epochs is 20-30.</li>
<li>Steps per epoch - total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch, we take 100. Optimal number is defined by the formula <b>length(training_set)//batch_size</b>.</li>
<li>Validation steps - total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch, we take 40. Optimal is defined by the formula <b>length(validation_set)//batch_size</b>.</li>
</ul></p>
<p>When compiling the model, we also need to define metrics, optimizer and loss function. In our experiments we trained both models with <b>metric</b> "accuracy", <b>optimizer</b> Adam, <b>loss function</b> sparse categorical cross entropy, to compare which architecture is more preferrable when it comes to semantic image segmentation.</p> 

### Results
<p>It took 20 minutes to train FCNN and 26 minutes to train U-Net models. Using FCNN we generated the following semantic segmentation image:</p>

![fcnna](https://github.com/Hatter01/semantic_segmentation/blob/main/images/fprediction.png) 

<p>Using U-Net the semantic segmentation image looks like this:</p>

![fcnna](https://github.com/Hatter01/semantic_segmentation/blob/main/images/uprediction.png) 

<p>Below you can see accuracy results <b>[FCNN - U-Net]</b>:</p>

![fcnna](https://github.com/Hatter01/semantic_segmentation/blob/main/images/faccuracy.png) 
![uneta](https://github.com/Hatter01/semantic_segmentation/blob/main/images/uaccuracy.png)

<p>Now we observe the loss results <b>[FCNN - U-Net]</b>:</p>

![fcnnl](https://github.com/Hatter01/semantic_segmentation/blob/main/images/floss.png) 
![unetl](https://github.com/Hatter01/semantic_segmentation/blob/main/images/uloss.png)

<p>Besides, we tested 3 optimizers RMSprop, AdamW and Adam with learning_rate=0.001 (which is pretty classic) only on FCNN. The results are different in the begining, but later they became very similar. Below you see the comparison of accuracy and loss taken from the tensorboard.</p>

![comp_acc](https://github.com/Hatter01/semantic_segmentation/blob/main/images/opt_comp_acc.png)
![comp_loss](https://github.com/Hatter01/semantic_segmentation/blob/main/images/opt_comp_loss.png)

<p>When playing with fcn_testing.ipynb file, we save logs when train the model. They can be used in tensorboard. Write the following command in console:</p>

```
tensorboard dev upload --logdir ./logs --name "some_name"
```

## Libraries and tools
<p>Python libraries used in the project:</p>
<ul>
<li>os</li>
<li>numpy</li>
<li>tensorflow</li>
<li>tensorboard</li>
<li>matplotlib</li>
<li>requests</li>
<li>tqdm</li>
<li>zipfile</li>
<li>datetime</li>
</ul>

<p>Tools:</p>
<ul>
<li>Git</li>
<li>Tensorboard.dev</li>
</ul>

## Runtime environment
<p>The project was designed in Visual Studio Code. Most functions are in .py format, but the process of training networks is done in .ipynb file.</p>

## Bibliography
```
@misc{xia2019publ,
  title={Cooperative Semantic Segmentation and Image Restoration in Adverse Environmental Conditions},
  author={Xia, Weihao and Cheng, Zhanglin and Yang, Yujiu and Xue, Jing-Hao},
  publisher={arXiv},
  doi={10.48550/ARXIV.1911.00679},
  url={https://arxiv.org/abs/1911.00679},
  keywords={Computer Vision and Pattern Recognition (cs.CV), Image and Video Processing (eess.IV), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  year={2019},
  copyright={arXiv.org perpetual, non-exclusive license}
}

@misc{barla2023guide,
  title={The Beginner’s Guide to Semantic Segmentation},
  author={Nilesh Barla},
  howpublished="\url{https://www.v7labs.com/blog/semantic-segmentation-guide#h2}",
  year={2023},
  note={Accessed: 2023-01-26}
}

@inproceedings{zhou2017scene,
  title={Scene Parsing through ADE20K Dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}

@misc{jordan2018blog,
  title={Evaluating image segmentation models},
  author={Jeremy Jordan},
  howpublished="\url{https://www.jeremyjordan.me/evaluating-image-segmentation-models/}",
  year={2018},
  note={Accessed: 2023-01-28}
}

@misc{tensor2015guide,
  title={Better performance with the tf.data API},
  author={Tensorflow},
  howpublished="\url{https://www.tensorflow.org/guide/data_performance#caching}",
  year={2015},
  note={Accessed: 2023-01-27}
}

@inproceedings{jadon2020survey,
  title={A survey of loss functions for semantic segmentation},
  author={Jadon, Shruti},
  booktitle={2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
```
