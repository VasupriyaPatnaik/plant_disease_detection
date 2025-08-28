from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
model = load_model('model.h5')  # Use local model path

app.config['DEBUG'] = False
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12MB max upload size
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(225, 225)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def classify_image(img_path, model):
    input_image = preprocess_image(img_path)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions, axis=1)
    class_names = ['Healthy', 'Powdery', 'Rust']
    predicted_class_label = class_names[predicted_class_index[0]]
    return predicted_class_label, predictions[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'result': "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'result': "No selected file"})

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)


            predicted_class_label, probabilities = classify_image(file_path, model)

            # Prepare probs_list for template: [(label, value, percent), ...]
            class_names = ['Healthy', 'Powdery', 'Rust']
            probs_list = []
            for idx, label in enumerate(class_names):
                val = float(probabilities[idx])
                pct = round(val * 100, 2)
                probs_list.append((label, val, pct))

            original_img = cv2.imread(file_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            os.remove(file_path)

            img_str = cv2.imencode('.jpg', original_img)[1].tobytes()
            img_base64 = "data:image/jpeg;base64," + base64.b64encode(img_str).decode('utf-8')

            from datetime import datetime
            current_year = datetime.now().year

            return render_template(
                'result.html',
                prediction=predicted_class_label,
                probs_list=probs_list,
                image=img_base64,
                current_year=current_year
            )

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            from flask import flash, redirect, url_for
            flash(f"Error processing file: {str(e)}")
            return redirect(url_for('home'))

    else:
        from flask import flash, redirect, url_for
        flash("File type not allowed. Please upload a JPG or PNG image.")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
