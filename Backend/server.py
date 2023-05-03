import cv2
from flask import Flask, render_template, Response, jsonify

from test_model_predictor import *


app = Flask(__name__, template_folder = './template')

camera = cv2.VideoCapture(0)

model = create_model()

compile_model(model)

model.load_weights('./Model/my_model_KJSCE.h5')

sequence = []
sentence = []
predictions = []

def gen_frames():  

    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break

        else:
            image = model_predictor(model, frame, sequence, sentence, predictions)

            ret, buffer = cv2.imencode('.jpeg', image)
            frame = buffer.tobytes()
            
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('./index.html', sentence = sentence)


# Used by frontend to fetch video frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Used by fontend to fetch sentences
@app.route('/sentence')
def get_sentences():
    return jsonify(sentence = sentence)


if __name__=='__main__':
    app.run(debug=True)


camera.release()
cv2.destroyAllWindows()