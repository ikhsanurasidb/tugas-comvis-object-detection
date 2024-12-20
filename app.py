from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diupload'
        
        file = request.files['file']
        if file.filename == '':
            return 'Tidak ada file yang dipilih'
        
        if file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = os.path.join(UPLOAD_FOLDER, f'input_{timestamp}.jpg')
            output_path = os.path.join(RESULT_FOLDER, f'result_{timestamp}.jpg')
            
            file.save(input_path)
            
            results = model.predict(input_path, save=False)
            
            result = results[0]
            
            img = cv2.imread(input_path)
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{result.names[cls]} {conf:.2f}'
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, img)
            
            return render_template('result.html', 
                                input_image=f'uploads/input_{timestamp}.jpg',
                                output_image=f'results/result_{timestamp}.jpg')
    
    return render_template('upload.html')

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(RESULT_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)