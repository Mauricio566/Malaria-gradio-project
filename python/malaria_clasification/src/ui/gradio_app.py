import gradio as gr
import requests
import json
from PIL import Image
import io

import base64

# Configuration
API_URL = "http://localhost:8000"
#API_URL = "http://localhost:8080"

def base64_to_image(base64_str):
    """Convierte base64 a PIL Image"""
    if not base64_str:
        return None
    
    # Remover el prefijo data:image/png;base64,
    image_data = base64_str.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

#Function that receives an image from Gradio and sends it to our API
def predict_malaria(image):
    try:
        # Convert PIL Image to bytes8
        img_byte_arr = io.BytesIO()#Creates an empty "in-memory file"
        image.save(img_byte_arr, format='PNG')#Save the image IN the memory box and Convert it to PNG format
        img_byte_arr.seek(0)
        
        # Send to API
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            pdf_path = result['result'].get('pdf_path')
            
            print("Claves en result:", result.keys())
            print("Claves en result['result']:", result['result'].keys())
            print("¿Tiene heatmap?", 'heatmap' in result['result'])
            print("¿Tiene overlay?", 'overlay' in result['result'])
            
            # Extract results
            prediction = result['result']['prediction']
            confidence = result['result']['confidence']
            class_id = result['result']['class_id']
            
            # Extraer imágenes
            heatmap_b64 = result['result'].get('heatmap')
            overlay_b64 = result['result'].get('overlay')
            
            # Convertir base64 a imágenes PIL
            heatmap_img = base64_to_image(heatmap_b64) if heatmap_b64 else None
            overlay_img = base64_to_image(overlay_b64) if overlay_b64 else None
            
            print("heatmap_b64 es None:", heatmap_b64 is None)
            print("overlay_b64 es None:", overlay_b64 is None)
            
            # Format response for Gradio
            status_emoji = "🦠" if prediction == "Infectado" else "✅"
            confidence_color = "red" if prediction == "Infectado" else "green"
            
            result_text = f"""
            ## Analysis Result
            ### {status_emoji} State: **{prediction}**
            ### Confidence: **{confidence}%**
            ### ID class: {class_id}"""
            
            return result_text,heatmap_img,overlay_img,pdf_path
            #return result_text

            
        else:
            return f" **Error server:** {response.status_code}\n\nDetail: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "**Connection error:** Unable to connect to the API server.\n\n💡 **Verify:**\n- The server is running\n- Correct URL: " + API_URL
        
    except Exception as e:
        return f" **unexpected error:** {str(e)}"

    
#Check if API is running
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health")
        #print("respuesta :) ",response.status_code)
        if response.status_code == 200:
            return "🟢Connected API "
            #return f"🟢Connected API {response.status_code}"

        else:
            return "🟡API with problems"
    except:
        return "🔴 API Disconnected "
        
    

# Create Gradio Interface
with gr.Blocks(
    title="Malaria Detection System",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    """
) as demo:
    
    # Header
    gr.Markdown(
        """# 🔬 Malaria Detection System
        """
    )
    
    # API Status
    with gr.Row():
        api_status = gr.Textbox(
            value=check_api_status(),
            label="server state",
            interactive=False
        )
        refresh_btn = gr.Button("update state", size="sm")
        refresh_btn.click(check_api_status, outputs=api_status)
    
    # Main Interface
    with gr.Row():
        with gr.Column(scale=1):
            # Image Input
            image_input = gr.Image(
                label="📁 Upload Blood Sample Image",
                type="pil",
                height=300
            )
            

            # Predict Button
            predict_btn = gr.Button(
                "🔍Analyze Sample", 
                variant="primary",
                size="lg"
            )
            
            # Clear Button
            clear_btn = gr.Button("🗑️ Clean", size="sm")
        
        with gr.Column(scale=1):
            # Results Output
            result_output = gr.Markdown(
                label="📋 Results",
                value="*Upload an image and press 'Analyze Sample' to see the results..*"
            )
            
            # Columna derecha: Outputs
        with gr.Column():
            
            with gr.Row():
                heatmap_output = gr.Image(label="Mapa de Calor", show_label=True)
                overlay_output = gr.Image(label="Imagen + Heatmap", show_label=True)
    
                pdf_output = gr.File(label="📄 Descargar Reporte PDF")
   
    
    # Wire up the functions
    predict_btn.click(
        fn=predict_malaria,
        inputs=image_input,
        outputs=[result_output, heatmap_output, overlay_output,pdf_output]
        #outputs=[result_output]

    )
    
    clear_btn.click(
        fn=lambda: (None, "*Upload an image and press 'Analyze Sample' to see the results..*"),
        inputs=[],
        #outputs=[image_input, result_output]
        outputs=[image_input, result_output, heatmap_output, overlay_output] 
    )

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio interface...")
    
# run python gradio_app.py    
    demo.launch(
        server_name="0.0.0.0",  # Accessible from other devices on network
        server_port=7860,       # Default Gradio port
        #share=False,            # Set to True to create public link
        share=True,
        show_error=True
    )