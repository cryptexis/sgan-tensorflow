import numpy as np
from PIL import Image, ImageDraw
import os


def normalize_img(src_img):

    out = np.array(src_img)
    return out / 127.5 - 1


def rescale_img(tensor):

    img = np.array(tensor)
    img = (img+1)*127.5
    return np.uint8(img)


def save_tensor(tensor, filename):

    img = rescale_img(tensor)
    img = Image.fromarray(img)
    img.save(filename)


def save_tensor_rect(tensor, rect_list, filename, color):

    img = rescale_img(tensor)
    img = Image.fromarray(img)
    dr = ImageDraw.Draw(img)
    for rect in rect_list:
        dr.rectangle(rect, outline=color)
    img.save(filename)


def get_data(folder, spatial_size, batch_size):

    img_array = []
    files = os.listdir(folder)
    for file_name in files:

        img_file = os.path.join(folder, file_name)
        img_obj = Image.open(img_file)
        img_array.append(normalize_img(img_obj))

    # assumes 3 channels for an image
    data = np.zeros((batch_size, spatial_size, spatial_size, 3))

    for i in range(batch_size):
        random_index = np.random.randint(len(img_array))
        img = img_array[random_index]
        if spatial_size < img.shape[0] and spatial_size < img.shape[1]:
            h = np.random.randint(img.shape[0] - spatial_size)
            w = np.random.randint(img.shape[1] - spatial_size)
            out = img[h:h + spatial_size, w:w + spatial_size, :]
        else:
            out = img
        data[i] = out
    return data


