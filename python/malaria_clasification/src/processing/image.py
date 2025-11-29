#Image processing utilities for malaria detection

import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def overlay_heatmap(original_image, heatmap, alpha=0.4):
   
    # Redimensionar heatmap para coincidir con imagen original
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    
    # Convertir a mapa de colores (jet colormap)
    heatmap_colored = cm.jet(heatmap_resized)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Convertir PIL a numpy
    original_np = np.array(original_image)
    
    # Mezclar imágenes
    blended = cv2.addWeighted(original_np, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(blended)

def image_to_base64(image):
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def heatmap_to_image(heatmap,size=(224,224)):
   
    # Convertir a rango [0, 255]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Crear imagen PIL y convertir a RGB
    heatmap_pil = Image.fromarray(heatmap_uint8)
    heatmap_pil = heatmap_pil.resize(size, Image.NEAREST)
    #heatmap_pil = heatmap_pil.convert('RGB')
    # Convertir a colormap "hot" o "jet"
    heatmap_colored = cm.hot(np.array(heatmap_pil) / 255.0)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(heatmap_colored)
    

def prepare_visualization_data(original_image, heatmap):
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap min/max: {heatmap.min()}/{heatmap.max()}")
    print(f"Heatmap dtype: {heatmap.dtype}")

    try:
        # Crear overlay
        overlay_image = overlay_heatmap(original_image, heatmap)
        
        # Convertir heatmap a imagen
        heatmap_image = heatmap_to_image(heatmap)
        
        # Antes de convertir a base64:
        print(f"Heatmap image size: {heatmap_image.size}")
        print(f"Heatmap image mode: {heatmap_image.mode}")
        
        # Convertir todo a base64
        return {
            "original": image_to_base64(original_image),
            "heatmap": image_to_base64(heatmap_image),
            "overlay": image_to_base64(overlay_image)
        }
    
    except Exception as e:
        print(f"Error preparing visualization data: {e}")
        return {
            "original": image_to_base64(original_image),
            "heatmap": None,
            "overlay": None
        }