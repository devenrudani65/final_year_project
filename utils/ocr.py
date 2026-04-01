import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import re

# Set Tesseract path (change if different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(uploaded_file):

    images = convert_from_bytes(uploaded_file.read())
    text = ""

    for img in images:
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text += pytesseract.image_to_string(gray)

    return text


def extract_text_from_image(uploaded_file):

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return text


def extract_cbc_values(text):

    text = text.lower()

    parameters = {
        "hemoglobin": r"hemoglobin\s*[:\-]?\s*(\d+\.?\d*)",
        "wbc": r"wbc\s*[:\-]?\s*(\d+\.?\d*)",
        "rbc": r"rbc\s*[:\-]?\s*(\d+\.?\d*)",
        "platelets": r"platelets?\s*[:\-]?\s*(\d+\.?\d*)",
        "mcv": r"mcv\s*[:\-]?\s*(\d+\.?\d*)",
        "mch": r"mch\s*[:\-]?\s*(\d+\.?\d*)",
        "mchc": r"mchc\s*[:\-]?\s*(\d+\.?\d*)",
        "neutrophils": r"neutrophils?\s*[:\-]?\s*(\d+\.?\d*)",
        "lymphocytes": r"lymphocytes?\s*[:\-]?\s*(\d+\.?\d*)",
        "monocytes": r"monocytes?\s*[:\-]?\s*(\d+\.?\d*)",
        "eosinophils": r"eosinophils?\s*[:\-]?\s*(\d+\.?\d*)",
        "basophils": r"basophils?\s*[:\-]?\s*(\d+\.?\d*)"
    }

    extracted = {}

    for key, pattern in parameters.items():

        match = re.search(pattern, text)

        if match:
            extracted[key] = float(match.group(1))

    return extracted