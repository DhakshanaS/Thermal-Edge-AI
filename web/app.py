import base64, cv2, numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Note: Loading a raw trtexec engine via Ultralytics relies on a UnicodeDecodeError fallback 
# in their loader. For full production, use native TensorRT Python bindings.
model = YOLO('models/thermal_cpp.engine', task='detect')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    encoded = data.split(",", 1)[1]
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    annotated = model(frame, verbose=False)[0].plot()
    b64_img = base64.b64encode(cv2.imencode('.jpg', annotated)[1]).decode('utf-8')
    return jsonify({'annotated_image': b64_img})
