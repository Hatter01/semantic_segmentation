import os
import requests
import zipfile
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Main downloading function
def download(url, path=None):
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    if not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)
    return fname

# Download ADE20K
def download_set(path):
    if not os.path.exists(path):
        os.mkdir(path)
    urls = ['http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
            'http://data.csail.mit.edu/places/ADEchallenge/release_test.zip',]
    download_dir = os.path.join(path, 'downloads')
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    for url in urls:
        fname = download(url, path=download_dir)
        with zipfile.ZipFile(fname, "r") as zip_ref:
            zip_ref.extractall(path=path)

# Load an image and its annotation (mask) and returning a dictionary.
def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels = 1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return {'image': image, 'mask': mask}

# Image visualization
def display(imgs):
    plt.figure(figsize=(17, 17))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(imgs[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset, model, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])