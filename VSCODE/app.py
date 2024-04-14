from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import base64
from ultralytics import YOLO
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import dlib
from imutils import face_utils
import time
import asyncio
from pygame import mixer

app = Flask(__name__)


# Load YOLO model
yolo_model = YOLO("best.pt")
yolo_classNames = ['Not Drowsy', 'Drowsy']

# Load VGG16 model
vgg16_model = load_model('vgg16.h5')
vgg16_classes = ['Closed', 'Open', 'Yawn', 'No_Yawn']

# Load CNN model
cnn_model = load_model('drowiness.h5')

# Load InceptionV3 model
inception_model = load_model('inceptionv3_pretrained.h5')

# Load the InceptionV3 model
model = load_model('inceptionv3.h5')


vgg16 = None
i1 = None
i2 = None
yoloo = None
cnnn = None


def predict_image_inceptionv3_scratch(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    class_labels = ['Closed', 'Open', 'no_yawn', 'yawn']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    
    if predicted_class in ["Open", "no_yawn"]:
        return "Not Drowsy"
    else:
        return "Drowsy"



# Preprocess image for InceptionV3 model
def preprocess_image_inceptionv3(image_path):
    img = cv2.imread(image_path)[...,::-1] # Read and convert to RGB
    resized_img = cv2.resize(img, (224, 224)) # Resize image to match model input size
    preprocessed_img = resized_img / 255.0 # Normalize pixel values
    return np.expand_dims(preprocessed_img, axis=0) # Add batch dimension

def predict_image(model, image_path):
    preprocessed_img = preprocess_image_inceptionv3(image_path)
    prediction = model.predict(preprocessed_img)
    class_index = np.argmax(prediction)
    return class_index, prediction[0][class_index]


# Preprocess image for CNN model
def preprocess_cnn_image(image):
    image_size = (80, 80)
    img = image.resize(image_size)
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

# Preprocess image for VGG16 model
def preprocess_vgg16_image(image):
    image_size = (224, 224)
    img = image.resize(image_size)
    img = np.array(img) / 255.0  # Normalize
    return img

# Process image for YOLO model
def process_yolo_image(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)

    detection_results = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Process each detected box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Append detection result as text
            cls = int(box.cls)
            currentClass = yolo_classNames[cls]
            conf = box.conf.item()
            text = f'{currentClass}: {conf:.2f}'
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detection_results.append(text)

    # Convert the annotated image to a base64 string
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode()
    return img_str, detection_results

def process_frame(frame):
    # Perform object detection using YOLO
    results = model(frame)

    detection_results = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Process each detected box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Annotate the frame with class and confidence
            
            cls = int(box.cls)
            currentClass = yolo_classNames[cls]
            conf = box.conf.item()
            text = f'{currentClass}: {conf:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append detection result as text
            detection_results.append(text)

    return frame, detection_results

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, detection_results = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
#rendering routes

mixer.init()
no_driver_sound = mixer.Sound('nodriver_audio.wav')
sleep_sound = mixer.Sound('sleep_sound.wav')
tired_sound = mixer.Sound('sleep_sound.wav')

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up/(2.0*down)

    # Checking if it is blinked
    if (ratio > 0.22):
        return 'active'
    else:
        return 'sleep'


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = compute(mouth[2], mouth[10])  # 51, 59
    B = compute(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = compute(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


(mStart, mEnd) = (49, 68)


async def tired():
    start = time.time()
    rest_time_start=start
    tired_sound.play()
    a = 0
    while (time.time()-start < 9):
        if(time.time()-rest_time_start>3):
            tired_sound.play()
        # cv2.imshow("USER",tired_img)
    tired_sound.stop()
    return


def detech():
    # status marking for current state
    sleep_sound_flag = 0
    no_driver_sound_flag = 0
    yawning = 0
    no_yawn = 0
    sleep = 0
    active = 0
    status = ""
    color = (0, 0, 0)
    no_driver=0
    frame_color = (0, 255, 0)
    # Initializing the camera and taking the instance
    cap = cv2.VideoCapture(0)

    # Give some time for camera to initialize(not required)
    time.sleep(1)
    start = time.time()
    no_driver_time=time.time()
    no_driver_sound_start = time.time()

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_frame = frame.copy()
        faces = detector(gray, 0)

        # detected face in faces array
        if faces:
         no_driver_sound_flag=0   
         no_driver_sound.stop()   
         no_driver=0  
         no_driver_time=time.time() 
        #  sleep_sound.stop()
         for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)
            # cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])
            mouth = landmarks[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR

            # Now judge what to do for the eye blinks

            if (mar > 0.70):
                sleep = 0
                active = 0
                yawning += 1
                status = "Drowsy"
                color = (255, 0, 0)
                frame_color = (255, 0, 0)
                sleep_sound_flag = 0
                sleep_sound.stop()

            elif (left_blink == 'sleep' or right_blink == 'sleep'):
                if (yawning > 20):
                    no_yawn += 1
                sleep += 1
                yawning = 0
                active = 0
                if (sleep > 5):
                    status = "Drowsy"
                    color = (0, 0, 255)
                    frame_color = (0, 0, 255)
                    if sleep_sound_flag == 0:
                        sleep_sound.play()
                    sleep_sound_flag = 1
            else:
                if (yawning > 20):
                    no_yawn += 1
                yawning = 0
                sleep = 0
                active += 1
                status = "Awake"
                color = (0, 255, 0)
                frame_color = (0, 255, 0)
                if active > 5:
                    sleep_sound_flag = 0
                    sleep_sound.stop()

            cv2.putText(frame, status, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if (time.time()-start < 60 and no_yawn >= 3):
                no_yawn = 0
                # print("tired")
                # asyncio.run(put_image(frame))
                # time.sleep(2)
                asyncio.run(tired())
            elif time.time()-start > 60:
                start = time.time()

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
        else:
            no_driver+=1
            sleep_sound_flag = 0
            sleep_sound.stop()
            if(no_driver>10):
              status="No Driver"
              color=(0,0,0)
            if time.time()-no_driver_time>5:
                if(no_driver_sound_flag==0):
                   no_driver_sound.play()
                   no_driver_sound_start=time.time()
                else:
                    if(time.time()-no_driver_sound_start>3):
                        no_driver_sound.play()
                        no_driver_sound_start=time.time()
                no_driver_sound_flag=1

        cv2.putText(frame, status, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("DRIVER (Enter q to exit)", frame)
        #cv2.imshow("68_POINTS", face_frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    no_driver_sound.stop()
    sleep_sound.stop()
    tired_sound.stop()
    cap.release()
    cv2.destroyAllWindows()   

@app.route("/open_camera")
def open():
    detech()
    print("open camera")
    return redirect("/")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/result1', methods=['POST'])
def result1():
    global yoloo
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            image_data, detection_results = process_yolo_image(file_path)
            yoloo = detection_results
            return render_template('result1.html', image_data=image_data, detection_results=detection_results)

@app.route('/result2', methods=['POST'])
def result2():
    global vgg16
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            image = Image.open(file)
            preprocessed_image = preprocess_vgg16_image(image)
            predictions = vgg16_model.predict(np.expand_dims(preprocessed_image, axis=0))
            predicted_class = np.argmax(predictions)
            predicted_class_name = vgg16_classes[predicted_class]
            if predicted_class_name.lower() == "open" or predicted_class_name.lower() == "no_yawn":
                prediction = "No Drowsiness detected"
            else:
                prediction = "Drowsiness detected"
            vgg16 = prediction
            return render_template('result2.html', filename=filename, prediction=prediction)

@app.route('/result3', methods=['POST'])
def result3():
    global cnnn
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            image = Image.open(file)
            processed_image = preprocess_cnn_image(image)
            result = cnn_model.predict(processed_image[np.newaxis, ...])
            predicted_label_index = np.argmax(result)
            if predicted_label_index in [0, 3]:
                prediction = 'Drowsiness Detected'
            else:
                prediction = 'No Drowsiness Detected'
            cnnn = prediction
            return render_template('result3.html', filename=filename, prediction=prediction)
        
@app.route('/result4', methods=['POST'])
def result4():
    global i1
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            class_index, confidence = predict_image(inception_model, file_path)
            class_label = "Drowsy" if class_index == 0 or class_index == 2 else "Not Drowsy"
            i1 = class_label
            return render_template('result4.html', result=f" {class_label}")
        
@app.route('/result5', methods=['POST'])
def result5():
    global i2
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            prediction = predict_image_inceptionv3_scratch(file_path)
            i2 = prediction
            return render_template('result5.html', prediction=prediction)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        # Check if file is included in the request
        if 'file' in request.files:
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                
                # Create dictionary with predictions
                results = {
                    'yolo_prediction': yoloo[0],
                    'vgg16_prediction': vgg16,
                    'cnn_prediction': cnnn,
                    'inception_pretrained_prediction': i1,
                    'inception_scratch_prediction': i2
                }
                
                return render_template('compare.html', results=results)
            else:
                # Handle case when no file is uploaded
                return "No file uploaded"
        else:
            # Handle case when file key is missing in the request
            return "File key missing in request"
    
    # Handle cases where it's a GET request
    return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(debug=True)


   