import os
import pytesseract
from pdf2image import convert_from_path
from shutil import copy2

def find_ipc_in_pdf(pdf_path, ipc_codes):
    try:
        whitelist = 'H0123456789'
        pages = convert_from_path(pdf_path, 300, first_page=1, last_page=1)
        text = pytesseract.image_to_string(pages[0], config=f'-c tessedit_char_whitelist={whitelist}')
        
        if text:
            for ipc_code in ipc_codes:
                if ipc_code.lower() in text.lower():
                    return ipc_code
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return None

def move_pdf_to_ipc_folder(pdf_path, ipc_code, output_folder):
    try:
        ipc_folder = os.path.join(output_folder, ipc_code)
        os.makedirs(ipc_folder, exist_ok=True)
        copy2(pdf_path, ipc_folder)
        print(f"Moved {pdf_path} to {ipc_folder}")
    except Exception as e:
        print(f"Error moving {pdf_path} to folder {ipc_folder}: {e}")

def scan_pdfs_in_folders(root_folder, ipc_codes, output_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                ipc_code = find_ipc_in_pdf(pdf_path, ipc_codes)
                if ipc_code:
                    print(f"Patent classified under IPC {ipc_code} found in: {pdf_path}")
                    move_pdf_to_ipc_folder(pdf_path, ipc_code, output_folder)

root_folder = 'C:/Users/SW6/Desktop/test'
ipc_codes = ['H01', 'H02', 'H03', 'H04', 'H05', 'H10']
output_folder = 'C:/Users/SW6/Downloads/test'
scan_pdfs_in_folders(root_folder, ipc_codes, output_folder)
