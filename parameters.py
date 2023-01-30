import os

root = os.getcwd()
def paths():
    dataset_path = root + "/ADEChallengeData2016/images/"
    training_data = "training/"
    val_data = "validation/"
    return dataset_path, training_data, val_data

def base():
    img_size = 128
    batch = 5
    buffer = 1000
    return img_size, batch, buffer