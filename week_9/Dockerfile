# Use the base Lambda Python 3.10 image
FROM agrigorev/zoomcamp-bees-wasps:v2

# Set environment variable for the model name
ENV MODEL_NAME=bees-wasps-v2.tflite

# Install required libraries
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install keras-image-helper
RUN pip install pillow

# Copy the Lambda function code
COPY lambda_function.py .

# Specify the command to run when the container starts (Lambda function)
CMD ["lambda_function.lambda_handler"]
