import os
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

def find_ipc_in_pdf(pdf_path, ipc_codes):
    try:
        whitelist = 'H0123456789'
        pages = convert_from_path(pdf_path, 300, first_page=1, last_page=1)
        text = pytesseract.image_to_string(pages[0], config=f'-c tessedit_char_whitelist={whitelist}')
        
        if text:
            for ipc_code in ipc_codes:
                if ipc_code.lower() in text.lower():
                    return True
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return False
def save_first_page(pdf_path, output_folder):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            writer = PdfWriter()
            first_page = reader.pages[0]
            writer.add_page(first_page)
            pdf_output_path = os.path.join(output_folder, os.path.basename(pdf_path) + "_page1.pdf")
            with open(pdf_output_path, 'wb') as output_file:
                writer.write(output_file)
            print(f"First page saved as PDF to: {pdf_output_path}")
    except Exception as e:
        print(f"Error saving first page of {pdf_path}: {e}")

def scan_pdfs_in_folders(root_folder, ipc_codes, output_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                if find_ipc_in_pdf(pdf_path, ipc_codes):
                    print(f"Patent classified under IPC H found in: {pdf_path}")
                    save_first_page(pdf_path, output_folder)

root_folder = 'C:/Users/SW6/Desktop/novo'
ipc_codes = ['H01', 'H02', 'H03', 'H04', 'H05', 'H10']
output_folder = 'C:/Users/SW6/Downloads/test'
scan_pdfs_in_folders(root_folder, ipc_codes, output_folder)