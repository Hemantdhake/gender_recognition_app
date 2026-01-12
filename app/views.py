# import os
# import cv2
# from app.face_recognition import face_Recognition_pipeline
# from flask import render_template, request
# import matplotlib.image as matimg


# UPLOAD_FOLDER = 'static/upload'

# def index():
#     return render_template('index.html')


# def app():
#     return render_template('app.html')


# def genderapp():
#     if request.method == 'POST':
#         f = request.files['image_name']
#         filename = f.filename
#         # save our image in upload folder
#         path = os.path.join(UPLOAD_FOLDER,filename)
#         f.save(path) # save image into upload folder
#         # get predictions
#         pred_image, predictions = face_Recognition_pipeline(path)
#         pred_filename = 'prediction_image.jpg'
#         cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
        
#         # generate report
#         report = []
#         for i , obj in enumerate(predictions):
#             gray_image = obj['roi'] # grayscale image (array)
#             eigen_image = obj['eig_img'].reshape(100,100) # eigen image (array)
#             gender_name = obj['prediction_name'] # name 
#             score = round(obj['score']*100,2) # probability score
            
#             # save grayscale and eigne in predict folder
#             gray_image_name = f'roi_{i}.jpg'
#             eig_image_name = f'eigen_{i}.jpg'
#             matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
#             matimg.imsave(f'./static/predict/{eig_image_name}',eigen_image,cmap='gray')
            
#             # save report 
#             report.append([gray_image_name,
#                            eig_image_name,
#                            gender_name,
#                            score])
            
        
#         return render_template('gender.html',fileupload=True,report=report) # POST REQUEST
            
    
    
#     return render_template('gender.html',fileupload=False) # GET REQUEST


import os
import cv2
import matplotlib.image as matimg
from flask import render_template, request
from app.face_recognition import face_Recognition_pipeline



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR → Flask_app/

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'upload')
PREDICT_FOLDER = os.path.join(BASE_DIR, 'static', 'predict')

# Create folders automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)



def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        file = request.files['image_name']
        filename = file.filename

        # SAVE IMAGE (ABSOLUTE PATH)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # ML PIPELINE
        pred_image, predictions = face_Recognition_pipeline(save_path)

        # SAVE PREDICTION IMAGE
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(os.path.join(PREDICT_FOLDER, pred_filename), pred_image)

        report = []
        for i, obj in enumerate(predictions):
            gray_img = obj['roi']
            eigen_img = obj['eig_img'].reshape(100, 100)

            gray_name = f'roi_{i}.jpg'
            eigen_name = f'eigen_{i}.jpg'

            matimg.imsave(os.path.join(PREDICT_FOLDER, gray_name), gray_img, cmap='gray')
            matimg.imsave(os.path.join(PREDICT_FOLDER, eigen_name), eigen_img, cmap='gray')

            report.append([
                gray_name,
                eigen_name,
                obj['prediction_name'],
                round(obj['score'] * 100, 2)
            ])

        return render_template('gender.html', fileupload=True, report=report)

    return render_template('gender.html', fileupload=False)
