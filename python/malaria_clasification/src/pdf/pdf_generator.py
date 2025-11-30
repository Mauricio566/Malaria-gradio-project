# pdf_generator.py
from fpdf import FPDF
from datetime import datetime
import base64
import io
from PIL import Image
import os

def create_simple_report(prediction_data, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Título
    pdf.cell(0, 10, 'Reporte Malaria', ln=True, align='C')
    pdf.ln(10)
    
    # Información básica
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Archivo: {filename}', ln=True)
    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', ln=True)
    pdf.ln(5)
    
    # Resultado
    pdf.set_font('Arial', 'B', 14)
    prediction = prediction_data.get('prediction', 'N/A')
    confidence = prediction_data.get('confidence', 0)
    
    pdf.cell(0, 10, f'Resultado: {prediction}', ln=True)
    pdf.cell(0, 10, f'Confianza: {confidence}%', ln=True)
    pdf.ln(10)
    
    # Función para agregar imagen desde base64
    def add_image_from_base64(base64_str, title, y_position):
        if not base64_str:
            return y_position
            
        try:
            # Decodificar base64
            if base64_str.startswith('data:'):
                image_data = base64_str.split(',')[1]
            else:
                image_data = base64_str
            
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Guardar temp
            temp_path = f"temp_{title.lower()}.png"
            img.save(temp_path)
            
            # Agregar al PDF
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, title, ln=True)
            pdf.image(temp_path, x=10, y=pdf.get_y(), w=80)
            
            # Limpiar temp
            os.remove(temp_path)
            
            return pdf.get_y() + 60  # Espacio para la imagen
            
        except Exception as e:
            print(f"Error agregando imagen {title}: {e}")
            return y_position
    
    # Agregar imágenes
    current_y = pdf.get_y()
    
    # Imagen overlay
    overlay_b64 = prediction_data.get('overlay')
    if overlay_b64:
        current_y = add_image_from_base64(overlay_b64, "Imagen con Analisis:", current_y)
        pdf.set_y(current_y + 10)
    
    # Crear directorio reports si no existe
    os.makedirs("reports", exist_ok=True) 
    # Guardar PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"reports\\reporte_{timestamp}.pdf"

    #pdf_path = f"python\\malaria_clasification\\src\\reports\\reporte_{timestamp}.pdf"
    pdf.output(pdf_path)    
    
    return pdf_path

# Función para llamar desde tu API
def generate_pdf(prediction_result, filename):
    try:
        return create_simple_report(prediction_result, filename)
    except Exception as e:
        print(f"Error: {e}")
        return None