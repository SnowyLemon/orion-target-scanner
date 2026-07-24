import cv2
import numpy as np
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles  # NEW: serves manifest.json + icons

# Import our existing scoring engine
from scan_target import analyze_orion_target

app = FastAPI()

# NEW: makes /static/icon-192.png etc. reachable. Create a "static" folder
# next to this file and put icon-192.png / icon-512.png in it.
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML & JavaScript interface served to mobile browsers
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orion Target Scanner</title>

    <!-- NEW: PWA manifest -->
    <link rel="manifest" href="/manifest.json">

    <!-- NEW: iOS-specific tags — these are what make "Add to Home Screen"
         behave like a standalone app instead of a bookmark on iPhone -->
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Target Scanner">
    <meta name="theme-color" content="#007aff">

    <style>
        body { font-family: -apple-system, sans-serif; text-align: center; padding: 20px; background: #f4f4f9; }
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
        button { background: #007aff; color: white; border: none; padding: 14px 24px; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer; margin-top: 15px; }
        input[type="file"] { display: none; }
        #results { margin-top: 20px; text-align: left; }
        img { width: 100%; border-radius: 8px; margin-top: 15px; }
        .score-row { display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 8px 0; }
        .total { font-size: 20px; font-weight: bold; color: #2c3e50; margin-top: 15px; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Orion Target Scanner</h2>
        <p>Take a photo of your 10m target sheet to calculate scores.</p>
        
        <label for="cameraInput">
            <button onclick="document.getElementById('cameraInput').click()">Snap Target Photo</button>
        </label>
        <!-- 'capture=environment' forces rear phone camera on iOS/Android -->
        <input type="file" id="cameraInput" accept="image/*" capture="environment" onchange="uploadImage()">
        
        <div id="loading" style="display:none; margin-top:15px;">Analyzing target...</div>
        
        <div id="results"></div>
        <img id="scoredImage" style="display:none;" />
    </div>

    <script>
        async function uploadImage() {
            const input = document.getElementById('cameraInput');
            if (!input.files[0]) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            document.getElementById('scoredImage').style.display = 'none';

            const formData = new FormData();
            formData.append('file', input.files[0]);

            try {
                const response = await fetch('/scan', { method: 'POST', body: formData });
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';

                if (data.error) {
                    alert('Error processing image: ' + data.error);
                    return;
                }

                let html = '<h3>Target Scores</h3>';
                data.scores.forEach((score, idx) => {
                    html += `<div class="score-row"><span>Target #${idx + 1}</span><span><b>${score}</b> (${data.distances[idx]} mm)</span></div>`;
                });
                html += `<div class="total">Total Score: ${data.total_score} / 109.0</div>`;
                
                document.getElementById('results').innerHTML = html;
                
                const img = document.getElementById('scoredImage');
                img.src = 'data:image/jpeg;base64,' + data.image_base64;
                img.style.display = 'block';

            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                alert('Upload failed: ' + err);
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the mobile web camera app."""
    return HTML_CONTENT

# NEW: serves the manifest file referenced in <head> above
@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")

@app.get("/ping")
def ping():
    return {"status": "awake"}

@app.post("/scan")
async def scan_target_endpoint(file: UploadFile = File(...)):
    """Receives image from mobile device, processes via OpenCV, returns JSON + preview."""
    try:
        # Save temporary image file
        temp_filename = "temp_mobile_input.jpg"
        contents = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(contents)
        
        # Run computer vision pipeline
        output_filename = "scored_mobile_output.jpg"
        scores, distances, total_score = analyze_orion_target(temp_filename, output_filename)
        
        # Read annotated image and convert to Base64 to show on phone screen
        with open(output_filename, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')
            
        return {
            "scores": scores,
            "distances": distances,
            "total_score": total_score,
            "image_base64": encoded_img
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Runs the server on all local network interfaces on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
