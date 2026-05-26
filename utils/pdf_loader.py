import PyPDF2
from typing import List, Dict, Any
import io

def extract_text_from_pdfs(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    """
    Extracts text from a list of uploaded PDF files.
    
    Args:
        uploaded_files: A list of Streamlit UploadedFile objects.
        
    Returns:
        A list of dictionaries containing filename, page number, and extracted text.
    """
    extracted_data = []
    
    for uploaded_file in uploaded_files:
        try:
            # We read the file content into a BytesIO object for PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            
            # Check if file is empty
            if len(pdf_reader.pages) == 0:
                print(f"Warning: {uploaded_file.name} appears to be empty.")
                continue
                
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text and text.strip():
                    extracted_data.append({
                        "filename": uploaded_file.name,
                        "page_number": page_num + 1,
                        "text": text.strip()
                    })
        except Exception as e:
            # Handle corrupted or unreadable files gracefully
            print(f"Error parsing {uploaded_file.name}: {e}")
            
    return extracted_data
