import os
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

MODEL_NAME = os.getenv('MODEL_NAME', 'bees-wasps-v2.tflite')

# Load the TF-Lite model
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

def download_image(url):
    with request.urlopen(url) as resp:
        img_data = resp.read()
    return Image.open(BytesIO(img_data))

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    target_size = (150, 150)  # Adjust the target size as needed
    preprocessed_img = prepare_image(img, target_size)

    # Convert the PIL image to a NumPy array
    image_array = np.array(preprocessed_img)

    # Preprocess the image array (assuming normalization is required)
    normalized_image = (image_array / 255.0).astype(np.float32)

    # Reshape the image
    desired_input_shape = (1, 150, 150, 3)
    reshaped_image = np.reshape(normalized_image, desired_input_shape)

    # Set the input tensor of the TF-Lite model
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(input_tensor_index)()[0] = reshaped_image

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor of the TF-Lite model
    output_tensor_index = interpreter.get_output_details()[0]['index']
    model_output = interpreter.tensor(output_tensor_index)()

    # Return the result (assuming it's a regression output)
    return float(model_output[0, 0])

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
