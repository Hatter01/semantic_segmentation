import tensorflow as tf
from utils import parse_image
from parameters import base, paths

# Providing seed to make results reproducible
seed = 1234
img_size, _, B = base()
dataset_path, training_data, val_data = paths()

@tf.function
def normalize(image: tf.Tensor, mask: tf.Tensor) -> tuple:
    image = tf.cast(image, tf.float32)/255.
    return image, mask

@tf.function
def transform_image(datapoint: dict) -> tuple:
    image = tf.image.resize(datapoint['image'], (img_size, img_size))
    mask = tf.image.resize(datapoint['mask'], (img_size, img_size))
    # Data augmentation
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    image, mask = normalize(image, mask)
    return image, mask

def dataset(batch, buffer):
    train_data = tf.data.Dataset.list_files(dataset_path + training_data + "*.jpg", seed = seed)
    train_data = train_data.map(parse_image)

    valid_data = tf.data.Dataset.list_files(dataset_path + val_data + "*.jpg", seed = seed)
    valid_data = valid_data.map(parse_image)

    dataset = {"train": train_data, "valid": valid_data}

    dataset['train'] = dataset['train'].map(transform_image, num_parallel_calls = tf.data.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size = buffer, seed = seed)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(batch)
    dataset['train'] = dataset['train'].prefetch(buffer_size = tf.data.AUTOTUNE)

    dataset['valid'] = dataset['valid'].map(transform_image)
    dataset['valid'] = dataset['valid'].repeat()
    dataset['valid'] = dataset['valid'].batch(batch)
    dataset['valid'] = dataset['valid'].prefetch(buffer_size = tf.data.AUTOTUNE)

    return dataset