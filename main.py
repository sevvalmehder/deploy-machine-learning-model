from flask import Flask, request, jsonify
import base64

from src.FashionMNISTClassifier import *

app = Flask(__name__)

model_path = "models/model.150-0.86.h5"

classifier = FashionMNISTClassifier()


@app.route('/predict', methods=['POST'])
def predict():
    base64_str = request.json['image']
    base64_str_decoded = base64.b64decode(base64_str)
    image = tf.io.decode_image(base64_str_decoded)
    # Convert from (28, 28, 3) to (1, 28, 28)
    image = (np.expand_dims(image[:, :, 0], 0))

    model = classifier.load_model(model_path)
    predicted_label = classifier.predict(image, model)

    return jsonify({'success': True, "data": predicted_label}), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1410)
