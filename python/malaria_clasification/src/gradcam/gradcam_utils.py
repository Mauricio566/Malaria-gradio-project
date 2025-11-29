#Grad-CAM utilities for explainable AI in malaria detection

import torch
import torch.nn.functional as F
import numpy as np

#Generate Grad-CAM to visualize which parts of the image are important
#image_tensor [1,C,H,W]
def generate_gradcam(model, image_tensor, target_class=None):
    target_layer = _find_target_layer(model)#  Find the last convolutional layer
    
    if target_layer is None:
        return _create_simple_gradient_map(model, image_tensor, target_class)
    
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Registrar hooks
    backward_handle = target_layer.register_backward_hook(backward_hook)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        # Forward pass
        model.zero_grad()
        output = model(image_tensor)
        
        # Usar clase predicha si no se especifica
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Obtener gradientes y activaciones
        gradients_tensor = gradients[0]
        activations_tensor = activations[0]
        
        # Calcular pesos (promedio global de gradientes)
        weights = torch.mean(gradients_tensor, dim=(2, 3))
        
        # Generar CAM
        cam = torch.zeros(activations_tensor.shape[2:], dtype=torch.float32)
        for i in range(len(weights[0])):
            cam += weights[0, i] * activations_tensor[0, i, :, :]
        
        # Aplicar ReLU y normalizar
        cam = F.relu(cam)
        cam = _normalize_cam(cam)
        
        return cam.detach().numpy()
        
    finally:
        # Limpiar hooks
        backward_handle.remove()
        forward_handle.remove()

def _find_target_layer(model):
    """
    Encuentra la última capa convolucional del modelo
    """
    target_layer = None
    
    # Buscar por diferentes arquitecturas comunes
    if hasattr(model, 'features'):
        # VGG-style
        target_layer = model.features[-1]
    elif hasattr(model, 'layer4'):
        # ResNet-style
        target_layer = model.layer4[-1]
    else:
        # Buscar manualmente la última capa Conv2d
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    
    return target_layer

def _create_simple_gradient_map(model, image_tensor, target_class=None):
    """
    Método fallback usando gradientes de entrada
    """
    image_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Obtener gradientes respecto a la entrada
    gradients = image_tensor.grad.data.abs()
    
    # Tomar máximo a través de canales de color
    cam = torch.max(gradients[0], dim=0)[0]
    
    # Normalizar
    cam = _normalize_cam(cam)
    
    return cam.numpy()

def _normalize_cam(cam):
    """
    Normaliza el CAM a rango [0, 1]
    """
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam