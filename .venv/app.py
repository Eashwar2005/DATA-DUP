from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import os
import random
import zipfile
import shutil

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================================
# CSV DATA DUPLICATION (UNCHANGED FROM YOUR WORKING VERSION)
# =========================================================

def expand_similar_dataset(df, target_rows):
    original_rows = len(df)
    synthetic_rows = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns

    while original_rows + len(synthetic_rows) < target_rows:
        base_row = df.sample(1).iloc[0].copy()

        for col in numeric_cols:
            std = df[col].std()
            if std > 0:
                base_row[col] += np.random.normal(0, std * 0.2)

        for col in categorical_cols:
            if random.random() < 0.3:
                base_row[col] = df[col].sample(1).values[0]

        synthetic_rows.append(base_row)

    expanded_df = pd.concat([df, pd.DataFrame(synthetic_rows)])
    return expanded_df.reset_index(drop=True)

# ==========================
# IMAGE AUGMENTATION LOGIC
# ==========================

def augment_image(img):
    if random.random() < 0.5:
        img = img.rotate(random.randint(-15, 15))

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    return img

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/csv", methods=["GET", "POST"])
def csv_page():
    if request.method == "POST":
        csv_file = request.files["csv_file"]
        target_rows = int(request.form["target_rows"])

        base_name = os.path.splitext(csv_file.filename)[0]
        input_path = os.path.join(UPLOAD_FOLDER, csv_file.filename)
        csv_file.save(input_path)

        df = pd.read_csv(input_path)
        expanded_df = expand_similar_dataset(df, target_rows)

        output_name = f"{base_name}_expanded.csv"
        output_path = os.path.join(UPLOAD_FOLDER, output_name)
        expanded_df.to_csv(output_path, index=False)

        return send_file(output_path, as_attachment=True)

    return render_template("csv.html")

@app.route("/image", methods=["GET", "POST"])
def image_page():
    if request.method == "POST":
        zip_file = request.files["zip_file"]
        target_images = int(request.form["target_images"])

        work_id = str(random.randint(10000, 99999))
        extract_dir = os.path.join(UPLOAD_FOLDER, f"extract_{work_id}")
        output_dir = os.path.join(UPLOAD_FOLDER, f"augmented_{work_id}")

        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        zip_path = os.path.join(extract_dir, zip_file.filename)
        zip_file.save(zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        images = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    images.append(os.path.join(root, f))

        if not images:
            return "No images found in ZIP", 400

        for i in range(target_images):
            img_path = random.choice(images)
            img = Image.open(img_path).convert("RGB")
            img = augment_image(img)
            img.save(os.path.join(output_dir, f"img_{i}.jpg"))

        output_zip = os.path.join(UPLOAD_FOLDER, f"augmented_images_{work_id}.zip")
        with zipfile.ZipFile(output_zip, "w") as zipf:
            for file in os.listdir(output_dir):
                zipf.write(os.path.join(output_dir, file), file)

        shutil.rmtree(extract_dir)
        shutil.rmtree(output_dir)

        return send_file(output_zip, as_attachment=True)

    return render_template("image.html")

if __name__ == "__main__":
    app.run(debug=True)
