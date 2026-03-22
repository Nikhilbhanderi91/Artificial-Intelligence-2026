# ==========================================
# AI Answer Sheet Evaluation using TrOCR
# ==========================================

import os
import re
import cv2
import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# STEP 1: CREATE FOLDERS
# ==========================================

os.makedirs("images", exist_ok=True)
os.makedirs("extracted_text", exist_ok=True)

# ==========================================
# STEP 2: PDF → IMAGE
# ==========================================

pdf_path = "input_pdf/student_answer.pdf"
images = convert_from_path(pdf_path)

for i, img in enumerate(images):
    img.save(f"images/page_{i}.jpg", "JPEG")

print("✅ PDF converted to image")

# ==========================================
# STEP 3: IMAGE PREPROCESSING
# ==========================================

img = cv2.imread("images/page_0.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)

# Improve contrast
gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

# Adaptive threshold
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

cv2.imwrite("images/cleaned.jpg", thresh)
print("✅ Image preprocessed")

# ==========================================
# STEP 4: HANDWRITTEN TEXT EXTRACTION (TrOCR)
# ==========================================

print("⏳ Loading TrOCR model (first time may take time)...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

image = Image.open("images/cleaned.jpg").convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n📝 Raw Extracted Text:\n")
print(generated_text)

# ==========================================
# STEP 5: TEXT CLEANING
# ==========================================

clean_text = re.sub(r'[^a-zA-Z\s]', '', generated_text)
clean_text = clean_text.lower()
clean_text = re.sub(r'\s+', ' ', clean_text)

print("\n🧹 Cleaned Text:\n")
print(clean_text)

with open("extracted_text/student.txt", "w") as f:
    f.write(clean_text)

# ==========================================
# STEP 6: TEXT SIMILARITY
# ==========================================

model_answer = """
LAN is local area network used for small area like office.
MAN is metropolitan area network used for city communication.
WAN is wide area network used for large geographical area.
"""

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform([model_answer, clean_text])

similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

print("\n🎯 Similarity Score:", similarity)

marks = similarity * 10
print("📝 Suggested Marks (Out of 10):", round(marks, 2))