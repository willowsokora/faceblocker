import os
import uuid
from flask import Flask, flash, request, send_file, abort
from werkzeug.utils import secure_filename
import cv2 as cv
import numpy as np

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'input')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
VIDEO_EXTENSIONS = {'mov', 'mp4', 'avi', 'wmv', 'flv', 'mkv', 'webm', 'm4v'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

app = Flask(__name__)

def is_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def is_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

def anonymize_face(image, blocks=5):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype='int')
    ySteps = np.linspace(0, h, blocks + 1, dtype='int')
    
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            roi = image[startY: endY, startX:endX]
            (B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
            cv.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    
    return image

def find_and_blur(bw, color):
    faceClassifier = cv.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml'))
    faces = faceClassifier.detectMultiScale(bw, 1.1, 4)
    for (x, y, w, h) in faces:
        color[y:y+h, x:x+w] = anonymize_face(color[y:y+h, x:x+w])
    profileClassifier = cv.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_profileface.xml'))
    profiles = profileClassifier.detectMultiScale(bw, 1.1, 4)
    for (x, y, w, h) in profiles:
        color[y:y+h, x:x+w] = anonymize_face(color[y:y+h, x:x+w])
    return color

@app.route('/blockfaces', methods=['POST'])
def block_faces():
    if 'video' not in request.files:
        return 'Invalid request'
    file = request.files['video']
    if file and is_video(file.filename):
        file_id = str(uuid.uuid4())
        filename = secure_filename(file_id + file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        capture = cv.VideoCapture(filepath)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output_filepath = os.path.join(OUTPUT_FOLDER, file_id + '.mp4')
        writer = None
        while (capture.isOpened()):
            ret, color = capture.read()
            if ret:
                if writer is None:
                    writer = cv.VideoWriter(output_filepath, fourcc, 30, (1280, 720))
                bw = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
                blurred = find_and_blur(bw, color)
                writer.write(cv.resize(blurred, (1280, 720)))
            else:
                break
        capture.release()
        writer.release()
        os.remove(filepath)
        return {
            'type': 'video',
            'extension': 'mp4',
            'id': file_id
        }
    elif file and is_image(file.filename):
        file_id = str(uuid.uuid4())
        filename = secure_filename(file_id + file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        image = cv.imread(filepath)
        bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = find_and_blur(bw, image)
        output_filepath = os.path.join(OUTPUT_FOLDER, file_id + '.png')
        cv.imwrite(output_filepath, blurred)
        os.remove(filepath)
        return {
            'type': 'image',
            'extension': 'png',
            'id': file_id
        }
    else:
        abort(500)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
app.run(host='0.0.0.0', port='5000')