import torch
from torchvision import transforms
from PIL import Image

from model_architecture import create_model

model = create_model()

#Returns a dictionary (OrderedDict) where the keys are the 
# layer names and the values are the tensors with the learned 
# weights and biases.

model.load_state_dict(# It receives that dictionary of weights and assigns them to the model architecture that you defined earlier.
    torch.load("malaria_detection_model.pth", map_location=torch.device("cpu"))
)

model.eval()

# Image transformations (same preprocessing as training)
transformation = transforms.Compose([
    transforms.Resize((224, 224)), # Change to the size used in your training
    transforms.ToTensor(),# Scale values of [0, 255] to [0, 1].
    transforms.Normalize([0.485, 0.456, 0.406],  # Si usaste normalización
                         [0.229, 0.224, 0.225])
])

# inference
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")# upload image
    tensor_imagen = transformation(image).unsqueeze(0) #the transformation is applied and a dimension called batch is added at the beginning. 
    
#Disables gradient calculation, saving memory and speeding up prediction, because we don't need backpropagation.
    with torch.no_grad():
        output = model(tensor_imagen) # The image is passed through the model and an output tensor is obtained with the probabilities (or logits) for each class. Example: tensor([[2.1, -1.3]]) for 2 classes.
        prediction = torch.argmax(output, dim=1).item() # The index of the class with the highest value is selected .item() converts it from a tensor to a Python number (int).
        predicted_class = "Infectado" if prediction == 0 else "No infectado"
        return predicted_class # 0 - 1

#It is used so that the code within that block is only executed 
# if you run the file directly, and not when you import it from 
# another script.
if __name__ == "__main__":
    image_path = "D:\\malaria_inference_project\\uninfected2.png" 
    predicted_class = predict_image(image_path, model)
    print(f"predicted class: {predicted_class}")
