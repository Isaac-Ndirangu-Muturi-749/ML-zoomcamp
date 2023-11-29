import requests

def invoke_lambda_function(image_url):
    lambda_invoke_url = 'https://7qvw3yo7hj.execute-api.eu-west-3.amazonaws.com/test/predict'
    payload = {'url': image_url}

    response = requests.post(lambda_invoke_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    test_image_url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'
    result = invoke_lambda_function(test_image_url)
    print(result)
