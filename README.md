
## Plant Disease Detection Web App

A web application to detect plant leaf diseases (Healthy, Powdery, Rust) using a deep learning model.

---

### Features
- Upload a leaf image and get instant disease prediction.
- Shows probability for each class.
- Displays suggested remedy based on the prediction.
- Modern, responsive UI.

---

### Setup Instructions

1. **Clone the repository**
        ```
        git clone https://github.com/VasupriyaPatnaik/plant_disease_detection.git
        cd plant_disease_detection
        ```

2. **Create a Python virtual environment**
        ```
        python -m venv venv
        .\venv\Scripts\activate
        ```

3. **Install dependencies**
        ```
        pip install -r requirements.txt
        ```

4. **Obtain the trained model (`model.h5`)**
        - Download `model.h5` from the project author, a release asset, or train your own using `model.py`.
        - Place `model.h5` in the project root directory (same folder as `app.py`).
        - **Do not commit `model.h5` to git.** Add it to your `.gitignore` file to avoid large binary files in your repository.

5. **Run the app**
        ```
        python app.py
        ```
        - Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### Usage
- Upload a leaf image (JPG/PNG, max 10MB).
- Click **Predict** to see the result.
- The app will show the predicted class, probabilities, and a suggested remedy for the detected disease.

---

### Limitations
- The model is trained only on leaf images. If you upload a non-leaf image, it will still predict one of the disease classes.
- For best results, use clear, well-lit images of a single leaf.
- The app does not reject non-leaf images automatically.

---

### Improving the Model
- To handle non-leaf images, retrain the model with an additional "Unknown" or "Not a leaf" class.
- You can also use a pre-trained classifier to filter out non-leaf images before prediction.

---

### Folder Structure
```
plant_disease_detection/
├── app.py
├── model.py
├── model.h5
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html
│   └── result.html
```

---

### License
MIT
