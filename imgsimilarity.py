import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as kimage
import numpy as np
import os
import pandas as pd
import time
import requests
import base64

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

def is_base64(sb):
    try:
        if isinstance(sb, str):
            # If the string is a URL-safe base64 string, replace URL-specific characters
            sb_bytes = sb.encode('utf-8')
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Input must be a string or bytes.")
        
        # Add padding if needed
        missing_padding = len(sb_bytes) % 4
        if missing_padding:
            sb_bytes += b'=' * (4 - missing_padding)
        
        # Decode the base64 string
        base64.b64decode(sb_bytes, validate=True)
        return True
    except Exception:
        return False

def load_and_preprocess_image(image_path):
    img = kimage.load_img(image_path, target_size=(224, 224))
    img_array = kimage.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_path):

    image_path = load_urlimg(image_path)
    img = load_and_preprocess_image(image_path)
    features = model.predict(img)
    ff = features[0][0][0].flatten()
    # print(ff.shape)
    return ff

def image_similarity_from_feature(features1, features2):
    # features1 = extract_features(image1_path)
    # features2 = extract_features(image2_path)

    # print(features1.shape)
    
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity

def image_similarity(image1_path, image2_path):
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)

    # print(features1.shape)
    
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity


def imgsEmbedd(IMG_FLODER,features_imgs_file):
    image_extensions = ['.jpg', '.jpeg', '.png'] #'.gif', '.bmp']  # Add more extensions if needed
    
    image_files = [file for file in os.listdir(IMG_FLODER) if os.path.splitext(file)[1].lower() in image_extensions]

    E = []
    for f in [x for x in image_files]:
        print(f'{IMG_FLODER}/{f}')
        image_path = f'{IMG_FLODER}/{f}'
        e = extract_features(image_path)
        # print(e.shape)
        E.append(e)


    df = pd.DataFrame(E)
    df['ID'] = image_files
    df['ID'] = IMG_FLODER+'/' + df['ID']

    df.to_csv(features_imgs_file, index=False)

def findSimilarity(input_image,X):
    input_image_features = extract_features(input_image)
    S = []
    for i in range(X.shape[0]):
        features2 = X.iloc[i]
        s = image_similarity_from_feature(input_image_features, features2)
        S.append(s)
    df['similarity'] = S

    sorted_df = df.sort_values(by='similarity', ascending=False)
    sorted_df = sorted_df[sorted_df['similarity'] > THRESHOLD]


    result_ID = []
    result_corr = []
    for index, row in sorted_df.iterrows():
        result_ID.append(row['ID'])
        result_corr.append(row['similarity'])

    return result_ID,result_corr

def base64_to_jpeg(base64_string, output_file):
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Write the binary data to a file
    with open(output_file, "wb") as file:
        file.write(image_data)

def load_urlimg(url):
    # URL of the image
    # url = 'https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg'
    try:
        response = requests.get(url)
        image_path = "current_img.jpg"
        with open(image_path, "wb") as file:
            file.write(response.content)
        return image_path
    except:
        if is_base64(url):
            image_path = "current_img.jpg"
            base64_to_jpeg(url, image_path)

        return url


url = 'https://static.amarintv.com/images/upload/editor/source/BuM2023/389807.jpg'

# url = 'imgs/2.jpeg'
# e = list(extract_features(url))
# print(e)
# print(e.shape)

s = image_similarity(url, url)
print(s)




# #start ------------------------------------------------------
# IMG_FLODER = 'test_images'
# THRESHOLD = 0.755

# image_extensions = ['.jpg', '.jpeg', '.png'] #'.gif', '.bmp']  # Add more extensions if needed
# image_files = [file for file in os.listdir(IMG_FLODER) if os.path.splitext(file)[1].lower() in image_extensions]
# image_files = [f'{IMG_FLODER}/{x}' for x in image_files]

# #Embbed featurea from imgs floder
# features_imgs_file = 'imgsfeature.csv'
# imgsEmbedd(IMG_FLODER,features_imgs_file)

# #read feauture data
# df = pd.read_csv('imgsfeature.csv')

# X = df.drop(columns=['ID'])
# #end start ------------------------------------------------------

# print('IMG_FLODER:',IMG_FLODER)
# for input_image in image_files:
#     start_time = time.time()

#     a,b = findSimilarity(input_image,X)

#     print('='*30)
#     print('input_image:',input_image)
#     print('similatity:',a)
#     print('confidence:',b)
#     print(f'time(from {len(image_files)} imges)',time.time()-start_time)















# # IMG_FLODER = '../data2'
# # input_image = '../data2/img52.jpeg'

# input_image = 'test_images/img1_1.jpg'

# #Embbed featurea from imgs floder
# features_imgs_file = 'imgsfeature.csv'
# imgsEmbedd(IMG_FLODER,features_imgs_file)

# #read feauture data
# df = pd.read_csv('imgsfeature.csv')

# X = df.drop(columns=['ID'])

# a,b = findSimilarity(input_image,X)

# print('==========================')
# print('input_image:',input_image)
# print('similatity:',a,b)

