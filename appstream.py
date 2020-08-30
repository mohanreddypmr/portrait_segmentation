import streamlit as st
import altair as altc
import pandas as pd
import numpy as np
import os, urllib, cv2
from PIL import Image
import cv2
import matplotlib.pyplot as plt


import tensorflow as tf
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,UpSampling2D,Conv2DTranspose

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from skimage.transform import resize



from keras import backend as K


def main():

    st.title('Portrait Segmentation')
    st.sidebar.title("Portrait Segmentation:")
    st.sidebar.markdown('Portrait segmentation refers to the process of segmenting a person in an image from its background.')

    model=load_model()
    uploaded_file = st.file_uploader("Choose a image file")
    if uploaded_file is not None:
        st.success('Image uploaded successfully')
        image = Image.open(uploaded_file)

        image= image.resize((256,256))

    if uploaded_file:
        img=np.reshape(image,(1,256,256,3))
        pred=model.predict([img])

        pred=np.array(pred).reshape((256,256))

        pre=pred<0.5
        pre=pre.astype(int)
        pre=pre*255

        ind=np.where(pre==0)
        img=img.reshape((256,256,3))
        mask=img.copy()
        mask[ind[0],ind[1],:]=0

        im=mask
        agree = st.sidebar.checkbox("Add background")
        if agree:
            backg1=Image.open('background/backg1.jpeg')
            backg1= backg1.resize((256,256))
            backg2=Image.open('background/backg2.jpeg')
            backg2= backg2.resize((256,256))
            backg3=Image.open('background/backg3.jpeg')
            backg3= backg3.resize((256,256))
            backg4=Image.open('background/backg4.jpeg')
            backg4= backg4.resize((256,256))
            backg5=Image.open('background/backg5.jpeg')
            backg5= backg5.resize((256,256))
            backg6=Image.open('background/backg6.jpeg')
            backg6= backg6.resize((256,256))


            st.sidebar.image([backg1,backg2,backg3],caption=['bg1 Image','bg2 Image','bg3 Image'] ,width=80, use_column_width=False)
            st.sidebar.image([backg4,backg5,backg6],caption=['bg4 Image','bg5 Image','bg6 Image'] ,width=80, use_column_width=False)
            st.sidebar.markdown('')

            app_mode = st.sidebar.selectbox("Choose the background Image",["bg1 image", "bg2 image","bg3 image", "bg4 image","bg5 image", "bg6 image"])
            st.markdown('')
            st.markdown('')
            bp=True
            if app_mode=='bg1 image':
                ind1=np.where(pre!=0)
                backg=backg1.copy()
                backg=np.array(backg)
            if app_mode=='bg2 image':
                ind1=np.where(pre!=0)
                backg=backg2.copy()
                backg=np.array(backg)
            if app_mode=='bg3 image':
                ind1=np.where(pre!=0)
                backg=backg3.copy()
                backg=np.array(backg)
            if app_mode=='bg4 image':
                ind1=np.where(pre!=0)
                backg=backg4.copy()
                backg=np.array(backg)
            if app_mode=='bg5 image':
                ind1=np.where(pre!=0)
                backg=backg5.copy()
                backg=np.array(backg)
            if app_mode=='bg6 image':
                ind1=np.where(pre!=0)
                backg=backg6.copy()
                backg=np.array(backg)
            backg[ind1[0],ind1[1],:]=0
            im=backg+mask

        st.image([image,im],caption=['original image','modified image'],width=320, use_column_width=False)
        st.sidebar.markdown("succesfully add background")
@st.cache(allow_output_mutation=True)
def segnet():

    input1=Input((256,256,3))
    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(input1)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    c1=BatchNormalization()(conv2)
    drop1 = Dropout(0.1)(c1)
    pool1 =MaxPooling2D(pool_size=(2, 2))(drop1)

    conv1=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    c2=BatchNormalization()(conv2)
    drop2 = Dropout(0.1)(c2)
    pool2 =MaxPooling2D(pool_size=(2, 2))(drop2)

    conv1=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    c3=BatchNormalization()(conv3)
    drop3 = Dropout(0.1)(c3)
    pool3 =MaxPooling2D(pool_size=(2, 2))(drop3)

    conv1=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    c4=BatchNormalization()(conv3)
    drop4 = Dropout(0.1)(c4)
    pool4 =MaxPooling2D(pool_size=(2, 2))(drop4)

    conv1=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool4)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    c5=BatchNormalization()(conv3)
    drop5 = Dropout(0.1)(c5)
    pool5 =MaxPooling2D(pool_size=(2, 2))(drop5)



    up1 =Conv2D(1024,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(pool5))
    merge1 = concatenate([c5,up1], axis =3)

    conv1=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge1)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    batch3=BatchNormalization()(conv3)
    batch3 = Dropout(0.2)(batch3)


    up2 =Conv2D(512,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch3))
    merge2 = concatenate([c4,up2], axis =3)

    conv1=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge2)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    batch3=BatchNormalization()(conv3)
    batch3 = Dropout(0.2)(batch3)


    up3 =Conv2D(256,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch3))
    merge3 = concatenate([c3,up3], axis =3)

    conv1=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge3)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    batch3=BatchNormalization()(conv3)
    batch3 = Dropout(0.2)(batch3)


    up4 =Conv2D(128,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch3))
    merge4 = concatenate([c2,up4], axis =3)

    conv1=Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(merge4)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    batch2 = Dropout(0.2)(batch2)


    up5 =Conv2D(64,1, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch2))
    merge5 = concatenate([c1,up5], axis =3)

    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)


    output=Conv2D(1,(1,1),activation='sigmoid')(batch2)

    model=Model(input1,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy",tf.keras.metrics.MeanIoU(num_classes=2)])
    return model

@st.cache(allow_output_mutation=True)
def load_model():
    model=segnet()
    model.load_weights('best_model.h5')
    return model

if __name__=='__main__':
    page_bg_img = '''
    <style>
    body {
    background-image : url("https://www.coollimos4less.com/wp-content/uploads/2015/09/dark-gray-blue-3d-color-backgrounds-hd-wallpaper-2560-x-1600-1024x640.jpg");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    main()
