from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import joblib

import utils

app = Flask(__name__)

# Load rescaler
scaler = joblib.load('models/min_max_scaler.pkl')

# Load the model
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.load_model(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
checkpoint_path = 'models/best_model.pth'
model, optimizer, start_epoch, best_val_loss = utils.load_checkpoint(model, optimizer, device, path=checkpoint_path)

model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('run_model.html')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_path = os.path.join('uploads', file.filename)
        img.save(img_path)

        tensor = transform_image(img_bytes)
        with torch.no_grad():
            output = model(tensor)
            rating = round(output.item(), 1)  # Round to the first decimal place
        return jsonify({'rating': rating, 'image_url': img_path})

# Route for serving uploaded images
@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
