import cv2
import numpy as np
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from scan_target import analyze_orion_target

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orion Target Scanner</title>

    <link rel="manifest" href="/manifest.json">
    <link rel="apple-touch-icon" href="/static/icon-192.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Target Scanner">
    <meta name="theme-color" content="#343B41">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>
        :root{
            --paper:#F5F1E6;
            --paper-edge:#E4DEC9;
            --ink:#22262A;
            --ink-soft:#6B6F62;
            --gunmetal:#343B41;
            --gunmetal-deep:#262B2F;
            --brass:#B08541;
            --brass-deep:#8C6830;
        }
        *{ box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: 'Inter', sans-serif;
            color: var(--ink);
            background: var(--paper);
            padding: 32px 16px;
        }
        .card {
            position: relative;
            background: white;
            border: 1px solid var(--paper-edge);
            border-radius: 14px;
            padding: 28px 26px 22px;
            box-shadow: 0 10px 30px rgba(34,38,42,0.08);
            max-width: 480px;
            margin: auto;
        }

        .eyebrow { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
        .eyebrow-label { font-family: 'Oswald', sans-serif; font-weight: 600; font-size: 12px; letter-spacing: .18em; color: var(--brass-deep); }
        .eyebrow-rule { flex: 1; height: 1px; background: var(--paper-edge); }

        h1 { font-family: 'Oswald', sans-serif; font-weight: 600; font-size: 26px; margin: 0 0 6px; letter-spacing: .01em; }
        .lede { margin: 0 0 22px; color: var(--ink-soft); font-size: 14.5px; line-height: 1.5; }

        .capture-zone { position: relative; padding: 26px 14px; margin-bottom: 8px; }
        .bracket { position: absolute; width: 20px; height: 20px; border-color: var(--brass); border-style: solid; opacity: .55; }
        .tl { top: 0; left: 0; border-width: 2px 0 0 2px; }
        .tr { top: 0; right: 0; border-width: 2px 2px 0 0; }
        .bl { bottom: 0; left: 0; border-width: 0 0 2px 2px; }
        .br { bottom: 0; right: 0; border-width: 0 2px 2px 0; }

        button.snap-btn {
            display: block; width: 100%;
            background: var(--brass); color: white; border: none;
            padding: 14px; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 15.5px;
            border-radius: 8px; cursor: pointer; transition: background .15s ease;
        }
        button.snap-btn:hover { background: var(--brass-deep); }
        button.snap-btn:focus-visible { outline: 2px solid var(--gunmetal); outline-offset: 2px; }

        button.save-btn {
            display: block; width: 100%;
            background: none; color: var(--gunmetal); border: 1px solid var(--paper-edge);
            padding: 10px; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 13.5px;
            border-radius: 8px; cursor: pointer; margin-top: 10px; transition: background .15s ease;
        }
        button.save-btn:hover { background: var(--paper); }
        button.save-btn:focus-visible { outline: 2px solid var(--gunmetal); outline-offset: 2px; }

        input[type="file"] { display: none; }

        #loading { display: none; text-align: center; font-size: 13.5px; color: var(--ink-soft); margin-top: 14px; }

        #results { margin-top: 20px; }
        #results h3 { font-family: 'Oswald', sans-serif; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-soft); margin: 0 0 10px; font-weight: 500; }

        .score-row {
            display: grid; grid-template-columns: 26px 1fr auto;
            align-items: center; gap: 10px; padding: 9px 0;
            border-bottom: 1px solid var(--paper-edge);
        }
        .idx {
            width: 22px; height: 22px; border-radius: 50%;
            background: var(--paper); border: 1px solid var(--paper-edge);
            display: flex; align-items: center; justify-content: center;
            font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-soft);
        }
        .row-label { font-size: 14px; }
        .row-figures { text-align: right; font-family: 'IBM Plex Mono', monospace; }
        .row-score { font-size: 15px; font-weight: 500; display: block; }
        .row-dist { font-size: 11.5px; color: var(--ink-soft); border-bottom: 1px dotted var(--brass); padding-bottom: 1px; }

        .total-panel {
            margin-top: 16px; background: var(--gunmetal); border-radius: 10px;
            padding: 14px 18px; display: flex; justify-content: space-between; align-items: baseline;
        }
        .total-label { font-family: 'Oswald', sans-serif; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #c9cdd1; }
        .total-value { font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 500; color: white; }

        .photo-frame { margin-top: 22px; }
        .photo-caption { font-family: 'Oswald', sans-serif; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-soft); margin: 0 0 6px; font-weight: 500; }
        img { width: 100%; border-radius: 8px; border: 1px solid var(--paper-edge); display: block; }

        .credit { margin-top: 22px; text-align: center; font-size: 11px; color: var(--ink-soft); opacity: .7; }
    </style>
</head>
<body>
    <div class="card">
        <div class="eyebrow">
            <span class="eyebrow-label">ORION</span>
            <span class="eyebrow-rule"></span>
        </div>

        <h1>Target scanner</h1>
        <p class="lede">Photograph your 10m target sheet to score it instantly.</p>

        <div class="capture-zone">
            <div class="bracket tl"></div>
            <div class="bracket tr"></div>
            <div class="bracket bl"></div>
            <div class="bracket br"></div>
            <label for="cameraInput">
                <button class="snap-btn" onclick="document.getElementById('cameraInput').click()">Snap target photo</button>
            </label>
            <input type="file" id="cameraInput" accept="image/*" capture="environment" onchange="uploadImage()">
        </div>

        <div id="loading">Reading target&hellip;</div>

        <div id="results"></div>

        <div class="photo-frame" id="scoredFrame" style="display:none;">
            <p class="photo-caption">Scored target</p>
            <img id="scoredImage" />
            <button class="save-btn" onclick="saveImage('scoredImage', 'scored-target.jpg')">Save Score</button>
        </div>

        <div class="photo-frame" id="overlayFrame" style="display:none;">
            <p class="photo-caption">All shots overlay</p>
            <img id="overlayImage" />
            <button class="save-btn" onclick="saveImage('overlayImage', 'shots-overlay.jpg')">Save Overlay</button>
        </div>

        <p class="credit">&copy; Mengde Lin</p>
    </div>

    <script>
        async function uploadImage() {
            const input = document.getElementById('cameraInput');
            if (!input.files[0]) return;

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            document.getElementById('scoredFrame').style.display = 'none';
            document.getElementById('overlayFrame').style.display = 'none';

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

                let html = '<h3>Your scores</h3>';
                data.scores.forEach((score, idx) => {
                    const formattedScore = Number.isInteger(score) ? score.toFixed(1) : score;
                    html += `<div class="score-row">
                        <span class="idx">${idx + 1}</span>
                        <span class="row-label">Target #${idx + 1}</span>
                        <span class="row-figures">
                            <span class="row-score">${formattedScore}</span>
                            <span class="row-dist">${data.distances[idx]} mm</span>
                        </span>
                    </div>`;
                });

                const formattedTotal = Number.isInteger(data.total_score) ? data.total_score.toFixed(1) : data.total_score;
                html += `<div class="total-panel"><span class="total-label">Total score</span><span class="total-value">${formattedTotal}</span></div>`;

                document.getElementById('results').innerHTML = html;

                const img = document.getElementById('scoredImage');
                img.src = 'data:image/jpeg;base64,' + data.image_base64;
                document.getElementById('scoredFrame').style.display = 'block';

                const overlayImg = document.getElementById('overlayImage');
                overlayImg.src = 'data:image/jpeg;base64,' + data.overlay_image_base64;
                document.getElementById('overlayFrame').style.display = 'block';

            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                alert('Upload failed: ' + err);
            }
        }

        async function saveImage(imgElementId, filename) {
            const img = document.getElementById(imgElementId);
            const dataUrl = img.src;

            const res = await fetch(dataUrl);
            const blob = await res.blob();

            // Plain anchor download - triggers a direct file save, no share sheet.
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")

@app.get("/ping")
def ping():
    return {"status": "awake"}

@app.post("/scan")
async def scan_target_endpoint(file: UploadFile = File(...)):
    try:
        temp_filename = "temp_mobile_input.jpg"
        contents = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(contents)

        output_filename = "scored_mobile_output.jpg"
        overlay_filename = "scored_overlay.jpg"
        scores, distances, total_score = analyze_orion_target(temp_filename, output_filename, overlay_filename)

        with open(output_filename, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')

        with open(overlay_filename, "rb") as f:
            overlay_encoded = base64.b64encode(f.read()).decode('utf-8')

        return {
            "scores": scores,
            "distances": distances,
            "total_score": total_score,
            "image_base64": encoded_img,
            "overlay_image_base64": overlay_encoded
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)