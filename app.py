from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan gambar
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model YOLOv8
model = YOLO('best.pt')  # Ganti dengan path model Anda

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diupload'
        
        file = request.files['file']
        if file.filename == '':
            return 'Tidak ada file yang dipilih'
        
        if file:
            # Simpan file yang diupload
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = os.path.join(UPLOAD_FOLDER, f'input_{timestamp}.jpg')
            output_path = os.path.join(RESULT_FOLDER, f'result_{timestamp}.jpg')
            
            # Simpan gambar input
            file.save(input_path)
            
            # Lakukan prediksi
            results = model.predict(input_path, save=False)
            
            # Ambil hasil prediksi pertama
            result = results[0]
            
            # Baca gambar menggunakan OpenCV
            img = cv2.imread(input_path)
            
            # Plot hasil deteksi pada gambar
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{result.names[cls]} {conf:.2f}'
                
                # Gambar box dan label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Simpan hasil
            cv2.imwrite(output_path, img)
            
            return render_template('result.html', 
                                input_image=f'uploads/input_{timestamp}.jpg',
                                output_image=f'results/result_{timestamp}.jpg')
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)