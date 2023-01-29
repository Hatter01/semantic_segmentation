# Report

## Author
Andrii Kozar, student of Poznan University of Technology

## Problem
<p>The problem's name is Semantic Image Segmentation. In an image classification task, the network assigns a label (or class) to each input image. However, suppose you want to know the shape of that object, which pixel belongs to which object, etc. In this case, you need to assign a class to each pixel of the image — this task is known as segmentation. A segmentation model returns much more detailed information about the image. Image segmentation has many applications in medical imaging, self-driving cars and satellite imaging, just to name a few.</p>
<p>Once more, the goal of the semantic segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we’re predicting for every pixel in the image, this task is commonly referred to as dense prediction. The expected output in semantic segmentation are not just labels or bounding box parameters, but a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. Thus it is a pixel level image classification.</p>
<p>To solve Semantic Segmentation problem, we use neural networks.</p>

## Dataset
The dataset used in problem solving task is ADE20K. It is composed of more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. The dataset contains images with 

### Dataset stats used in project
<ul>The current version of the dataset contains:
  <li>22,210 images (20,210 for training and 2,000 for testing).</li>
  <li>150 categories (~classes)</li>
</ul>

## Architectures of NN
### U-Net

### FCN
## 
