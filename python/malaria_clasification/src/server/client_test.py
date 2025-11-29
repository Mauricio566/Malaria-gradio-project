#image_file = open(image_path, 'rb') open file
#files = {"file": image_file} Use file
#image_file.close()  Close file (manual)

import requests
import json
import os

# Base URL of your API
#your server address
API_URL = "http://localhost:8000"

def test_health_endpoint():
    try:
        response = requests.get(f"{API_URL}/health")#Returns: A Response object containing: The complete server response
        print(f"Status Code: {response.status_code}")# 200,404 etc
        print(f"Response: {response.json()}")# Convert answer to dict, JSON is text: '{"name": "John"}', Dictionary is a Python object: {"name": "John"}
        print()
        return response.status_code == 200# True if everything OK
    
    #If you cannot connect to the server → error
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the API.")
        return False

#Submit an image to the API for prediction
def predict_image(image_path):
    if not os.path.exists(image_path):# Does the file exist?
        print(f"Error: the image {image_path} does not exist")
        return None
    
    try:
        #Verify that the file exists before sending it
        with open(image_path, 'rb') as image_file: # binary mode, For non-text files (images, videos, etc.)
            files = {"file": image_file}
            response = requests.post(f"{API_URL}/predict", files=files)# An image is sent to the server at this specific path
        
        print(f"Status Code: {response.status_code}")
        
        #Extract data from the JSON returned by your API
        if response.status_code == 200:
            result = response.json()
            print(" Successful prediction:")
            print(f" file: {result['filename']}")# simple access.
            print(f" Result: {result['result']['prediction']}")# anidado access.
            print(f" Confidence: {result['result']['confidence']}%")
            print(f" ID class: {result['result']['class_id']}")
            return result
        else:
            print(f" prediction error:")
            print(f" {response.json()}")
            return None
    
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the API.")
        return None
    except Exception as e:
        print(f"unexpected error.: {e}")
        return None

#test multiple images
def test_multiple_images(image_paths):
    results = []# Empty list to save results
    
    for image_path in image_paths:# Iterate over each image path
        result = predict_image(image_path)
        if result:# if is true
            results.append(result)
        print("-" * 50)# -----... 50 
    
    return results

if __name__ == "__main__":
    test_images = [
        "D:\\malaria_inference_project\\uninfected2.png",
        #"uninfected2.png",
    
    ]
    #predictions
    results = test_multiple_images(test_images)
    
    # final summary
    for i, result in enumerate(results, 1):
        prediction = result['result']['prediction']
        confidence = result['result']['confidence']
        filename = result['filename']
        print(f"{i}. {filename}: {prediction} ({confidence}%)")
    
    print(f"tests completed.: {len(results)} tests completed.")