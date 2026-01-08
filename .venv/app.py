from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
            if np.random.rand() < 0.3:
                base_row[col] = df[col].sample(1).values[0]

        synthetic_rows.append(base_row)

    expanded_df = pd.concat([df, pd.DataFrame(synthetic_rows)])
    return expanded_df.reset_index(drop=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        csv_file = request.files["csv_file"]
        target_rows = int(request.form["target_rows"])

        # Original filename handling
        original_filename = os.path.splitext(csv_file.filename)[0]

        input_path = os.path.join(UPLOAD_FOLDER, csv_file.filename)
        csv_file.save(input_path)

        df = pd.read_csv(input_path)
        expanded_df = expand_similar_dataset(df, target_rows)

        output_filename = f"{original_filename}_expanded.csv"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        expanded_df.to_csv(output_path, index=False)

        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
