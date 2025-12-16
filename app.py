from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_data = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            # Convert FileStorage to BytesIO
            img_bytes = io.BytesIO(file.read())

            img = image.load_img(img_bytes, target_size=(160, 160))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            result = model.predict(img_array)[0][0]
            prediction = "Dog 🐶" if result > 0.5 else "Cat 🐱"

            # Send image back to frontend
            img_bytes.seek(0)
            img_data = img_bytes.read()

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
