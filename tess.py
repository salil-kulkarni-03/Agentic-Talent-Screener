from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = Image.open("C:/Users/kshri/OneDrive/Desktop/AI Recruiter/Tesseract_Example.png")

text = pytesseract.image_to_string(image)

print(text)