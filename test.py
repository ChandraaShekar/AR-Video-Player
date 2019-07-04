import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def read_img_from_dir(directory, query_shape):
    # query_shape is a tuple which contain the size (width, height) of query image
    # directory is your dir contain image you wanna find
    name_image = []
    shape = query_shape
    first = True
    for pics in os.listdir(directory):
        name_image.append("dataset/"+pics)
        image = Image.open("dataset/"+pics)
        image = image.resize(shape)
        image = np.array(image)
        image = np.reshape(image,(1,-1))
        if first:
            img_array = np.copy(image)
            first = False
        else:
            img_array = np.concatenate((img,array,image),axis=0)
    return name_image, img_array    

def find_by_knn(img, list_name, list_array):
    # image_query is path of your picture you wanna find
    # list_name and list_array is result of above function
    img = np.reshape(img,(1,-1))
    num_pics = list_array.shape[0]
    dists = np.zeros((num_pics,1))
    dists = list(np.sqrt(np.sum((list_array-img)**2,axis = 1)))
    idx = dists.index(max(dists))
    return list_name[idx]


directory = "dataset"
image_query = "q/out.jpg"
img = cv2.imread(image_query)
shape = img.shape[:2]
name_image, img_array = read_img_from_dir(directory,shape)
result = find_by_knn(img, name_image, img_array)
print(result)
