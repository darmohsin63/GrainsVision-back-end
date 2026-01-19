from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI(title="GrainsVision Analysis API", version="1.0.0")

# (Optional) allow browser calls directly; you can tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def analyze_image_bytes(image_bytes: bytes):
    # Decode image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    # ---- Minimal baseline "analysis" (placeholder) ----
    # Replace this section with your real OpenCV pipeline.
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Count connected components as a rough proxy for "grains"
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - th, connectivity=8)
    # label 0 is background
    total = max(0, num_labels - 1)

    # Dummy splits (replace with classification rules)
    foreign = int(total * 0.05)
    broken = int(total * 0.10)
    healthy = max(0, total - foreign - broken)

    # Dummy size stats (replace with measured contour sizes)
    avg_size_mm = 4.00
    min_size_mm = 2.50
    max_size_mm = 6.20

    def pct(x): 
        return round((x / total) * 100, 2) if total else 0.0

    return {
        "total_count": total,
        "healthy_count": healthy,
        "broken_count": broken,
        "foreign_material_count": foreign,
        "avg_size_mm": avg_size_mm,
        "min_size_mm": min_size_mm,
        "max_size_mm": max_size_mm,
        "healthy_percentage": pct(healthy),
        "broken_percentage": pct(broken),
        "foreign_material_percentage": pct(foreign),
        "meta": {"width": w, "height": h},
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = analyze_image_bytes(image_bytes)
    return result
    
