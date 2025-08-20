import pytesseract
import os

# Try to find Tesseract automatically
possible_paths = [
    r'D:\Anaconda\Tesseract-OCR\tesseract.exe'
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
]

for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break