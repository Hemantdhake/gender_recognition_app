import pandas as pd
import numpy as np
import pickle
import os
import cv2 as cv


harr = cv.CascadeClassifier("D:\\STUDY\\Programming\\project\\Model 1\\face_recognition_train_ml\\model\\haarcascade_frontalface_default.xml")
model_svm = pickle.load(open("D:\\STUDY\\Programming\\project\\Model 1\\face_recognition_train_ml\\model\\Face_Rec_SVMModel.pickle" , mode = 'rb'))

pca_models = pickle.load(open("D:\\STUDY\\Programming\\project\\Model 1\\face_recognition_train_ml\\model\\pca_dict.pickle" , mode = 'rb'))

model_pca = pca_models['pca'] # pca model
mean_face_arr = pca_models['mean_face'] # mean face

def face_Recognition_pipeline(filename , path = True):
    if path: # if the 
        # step-01: read image
        img = cv.imread(filename) # real time image array 
    else:
        img = filename # image array
    # step-02: convert into gray scale
    gray =  cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
    # step-03: crop the face (using haar cascase classifier)
    faces = harr.detectMultiScale(gray,1.5,3)
    predictions = []
    for x,y,w,h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = gray[y:y+h,x:x+w]
        
        # step-04: normalization (0-1)
        roi = roi / 255.0
        # step-05: resize images (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv.resize(roi,(100,100),cv.INTER_AREA)
        else:
            roi_resize = cv.resize(roi,(100,100),cv.INTER_CUBIC)
            
        # step-06: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000)
        # step-07: subtract with mean
        roi_mean = roi_reshape - mean_face_arr # subtract face with mean face
        # step-08: get eigen image (apply roi_mean to pca)
        eigen_image = model_pca.transform(roi_mean)
        # step-09 Eigen Image for Visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        # step-10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        
        # step-11: generate report
        text = "%s : %d"%(results[0],prob_score_max*100)
        # defining color based on results
        if results[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)
            
        cv.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv.putText(img,text,(x,y),cv.FONT_HERSHEY_PLAIN,3,(255,255,255),5)
        output = {
            'roi':roi,
            'eig_img': eig_img,
            'prediction_name':results[0],
            'score':prob_score_max
        }
        
        predictions.append(output)

    return img , predictions