// Global State variables
let uploadedFile = null;
let currentModality = 'florence'; // 'florence', 'depth', 'clip'
let originalImage = null; // Store Image object to redraw

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const canvasContainer = document.getElementById('canvas-container');
const imageCanvas = document.getElementById('image-canvas');
const ctx = imageCanvas.getContext('2d');
const depthOverlay = document.getElementById('depth-overlay');
const depthImg = document.getElementById('depth-img');

const navTabs = document.querySelectorAll('.nav-tab');
const modalityParams = document.querySelectorAll('.modality-params');
const florenceTaskSelect = document.getElementById('florence-task');
const refExpGroup = document.getElementById('ref-exp-group');
const florenceTextInput = document.getElementById('florence-text');
const clipCandidatesTextarea = document.getElementById('clip-candidates');

const btnRun = document.getElementById('btn-run');
const btnClear = document.getElementById('btn-clear');
const statLatency = document.getElementById('stat-latency');
const statusDot = document.querySelector('.status-dot');

const consoleTabs = document.querySelectorAll('.console-tab');
const consoleContents = document.querySelectorAll('.console-content');
const consoleVerdict = document.getElementById('console-verdict');
const rawJsonOutput = document.getElementById('raw-json-output');

// Colors for Annotations
const ANNOTATION_COLORS = [
    '#a78bfa', // Purple
    '#38bdf8', // Blue
    '#34d399', // Green
    '#f472b6', // Pink
    '#fbbf24', // Yellow
    '#f97316'  // Orange
];

/* Drag and Drop File Handlers */
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

dropZone.addEventListener('click', (e) => {
    // Prevent triggering browse dialog when clicking inside canvas/image
    if (e.target === dropZone || dropZone.contains(e.target) && !canvasContainer.contains(e.target)) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Invalid file format. Please upload an image file.');
        return;
    }
    
    uploadedFile = file;
    btnRun.removeAttribute('disabled');
    
    const reader = new FileReader();
    reader.onload = (event) => {
        originalImage = new Image();
        originalImage.onload = () => {
            // Set canvas drawing size
            imageCanvas.width = originalImage.width;
            imageCanvas.height = originalImage.height;
            
            // Draw image on canvas
            ctx.drawImage(originalImage, 0, 0);
            
            // Update viewports
            document.querySelector('.drop-prompt').style.display = 'none';
            canvasContainer.style.display = 'flex';
            depthOverlay.style.display = 'none';
        };
        originalImage.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

/* Tab Switching Modalities */
navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        navTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        currentModality = tab.dataset.modality;
        
        // Hide parameter columns
        modalityParams.forEach(panel => panel.classList.remove('active'));
        document.getElementById(`params-${currentModality}`).classList.add('active');
        
        // Reset overlay
        if (currentModality !== 'depth') {
            depthOverlay.style.display = 'none';
            if (originalImage) {
                resetCanvas();
            }
        } else if (depthImg.src) {
            depthOverlay.style.display = 'flex';
        }
    });
});

/* Florence Task Options Specific Selector Toggle */
florenceTaskSelect.addEventListener('change', () => {
    if (florenceTaskSelect.value === '<REFERRING_EXPRESSION_SEGMENTATION>') {
        refExpGroup.style.display = 'flex';
    } else {
        refExpGroup.style.display = 'none';
    }
});

/* Console Tab Navigation */
consoleTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        consoleTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        consoleContents.forEach(c => c.classList.remove('active'));
        document.getElementById(`console-${tab.dataset.console}`).classList.add('active');
    });
});

// Clear button
btnClear.addEventListener('click', () => {
    consoleVerdict.innerHTML = `
        <div class="placeholder-message">
            <i class="fa-solid fa-terminal"></i>
            <p>Upload an image and press "Run Inference" to inspect predictions.</p>
        </div>
    `;
    rawJsonOutput.textContent = '{ "status": "awaiting_input" }';
    statLatency.textContent = '-- ms';
    
    if (originalImage) {
        resetCanvas();
    }
});

function resetCanvas() {
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.drawImage(originalImage, 0, 0);
    depthOverlay.style.display = 'none';
}

/* Call Inference Engine */
btnRun.addEventListener('click', async () => {
    if (!uploadedFile) return;
    
    // Toggle Loading Indicators
    btnRun.setAttribute('disabled', 'true');
    statusDot.className = 'status-dot loading';
    statLatency.textContent = 'loading...';
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    let url = '';
    if (currentModality === 'florence') {
        url = '/api/analyze';
        formData.append('task', florenceTaskSelect.value);
        if (florenceTaskSelect.value === '<REFERRING_EXPRESSION_SEGMENTATION>') {
            formData.append('text_input', florenceTextInput.value || 'the object');
        }
    } else if (currentModality === 'depth') {
        url = '/api/depth';
    } else if (currentModality === 'clip') {
        url = '/api/similarity';
        formData.append('candidates', clipCandidatesTextarea.value || 'an image');
    }
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }
        
        const data = await response.json();
        
        // Render Raw JSON response
        rawJsonOutput.textContent = JSON.stringify(data, null, 2);
        
        // Show Latency Stats
        const latencyMs = Math.round(data.latency * 1000);
        statLatency.textContent = `${latencyMs} ms`;
        
        // Handle visualization overlays based on modality
        resetCanvas();
        renderResult(data);
        
    } catch (err) {
        console.error(err);
        statLatency.textContent = 'error';
        consoleVerdict.innerHTML = `
            <div class="caption-box" style="border-color: var(--color-neon-pink); background: rgba(239, 68, 68, 0.1);">
                <span style="color: var(--color-neon-pink);"><i class="fa-solid fa-triangle-exclamation"></i> Inference Failed</span>
                <p>${err.message}. Ensure the local FastAPI server is running.</p>
            </div>
        `;
    } finally {
        btnRun.removeAttribute('disabled');
        statusDot.className = 'status-dot online';
    }
});

/* Processing results to display */
function renderResult(data) {
    if (currentModality === 'depth') {
        // Render depth overlay image
        depthImg.src = data.depth_image;
        depthOverlay.style.display = 'flex';
        
        consoleVerdict.innerHTML = `
            <div class="caption-box">
                <span>Depth Metrics</span>
                <p>Depth prediction loaded successfully. Closer pixels are colorized brighter.
                <br><strong>Depth statistics:</strong> Mean value = ${data.stats.mean.toFixed(2)} [Min: ${data.stats.min.toFixed(2)}, Max: ${data.stats.max.toFixed(2)}]</p>
            </div>
        `;
        return;
    }
    
    if (currentModality === 'clip') {
        let rows = '';
        data.matches.forEach((match, index) => {
            const percent = (match.probability * 100).toFixed(1);
            rows += `
                <tr>
                    <td><strong>#${index + 1}</strong></td>
                    <td>${match.label}</td>
                    <td>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
                            <span style="font-family: var(--font-mono); font-weight: 500;">${percent}%</span>
                            <div style="flex: 1; max-width: 150px;">
                                <div class="prob-bar-container">
                                    <div class="prob-bar" style="width: ${percent}%;"></div>
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        consoleVerdict.innerHTML = `
            <table class="similarity-table">
                <thead>
                    <tr>
                        <th width="80px">Rank</th>
                        <th>Tag/Candidate</th>
                        <th>Probability Match</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;
        return;
    }
    
    if (currentModality === 'florence') {
        const task = florenceTaskSelect.value;
        const resultPayload = data.result[task];
        
        if (!resultPayload) {
            consoleVerdict.innerHTML = `<p style="color: var(--color-muted);">No structured results returned for task ${task}</p>`;
            return;
        }
        
        // Handle Object Detection and Dense Region Captioning (Drawing boxes)
        if (task === '<OD>' || task === '<DENSE_REGION_CAPTION>' || task === '<OCR_WITH_REGION>') {
            let boxes = [];
            let labels = [];
            
            if (task === '<OD>' || task === '<DENSE_REGION_CAPTION>') {
                boxes = resultPayload.bboxes || [];
                labels = resultPayload.labels || [];
            } else if (task === '<OCR_WITH_REGION>') {
                // OCR returnsquad boxes or text bboxes
                boxes = resultPayload.quad_boxes || [];
                labels = resultPayload.labels || [];
            }
            
            if (boxes.length === 0) {
                consoleVerdict.innerHTML = `<p style="color: var(--color-muted);">No objects detected in the image.</p>`;
                return;
            }
            
            // Draw on canvas
            drawBboxes(boxes, labels);
            
            // Output summary details to console list
            let boxesHtml = '';
            boxes.forEach((box, i) => {
                const color = ANNOTATION_COLORS[i % ANNOTATION_COLORS.length];
                const label = labels[i] || `Region #${i + 1}`;
                boxesHtml += `
                    <div class="prediction-box">
                        <div class="pred-left">
                            <span class="pred-color-tag" style="color: ${color};"></span>
                            <span class="pred-label">${label}</span>
                        </div>
                        <span class="pred-coords">[${box.join(', ')}]</span>
                    </div>
                `;
            });
            consoleVerdict.innerHTML = boxesHtml;
            
        } else if (task === '<REFERRING_EXPRESSION_SEGMENTATION>') {
            // Draw polygons for segmentations
            const polygons = resultPayload.polygons || [];
            const labels = resultPayload.labels || [];
            
            if (polygons.length === 0) {
                consoleVerdict.innerHTML = `<p style="color: var(--color-muted);">No segments found matching the expression.</p>`;
                return;
            }
            
            drawPolygons(polygons, labels);
            
            consoleVerdict.innerHTML = `
                <div class="caption-box">
                    <span>Segmentation Mask Rendered</span>
                    <p>Successfully rendered <strong>${polygons.length}</strong> masks for the query target <strong>"${florenceTextInput.value}"</strong>.</p>
                </div>
            `;
            
        } else if (task === '<CAPTION>' || task === '<DETAILED_CAPTION>' || task === '<MORE_DETAILED_CAPTION>' || task === '<OCR>') {
            // Handle plain-text caption or OCR
            const textValue = typeof resultPayload === 'string' ? resultPayload : JSON.stringify(resultPayload);
            const titleLabel = task === '<OCR>' ? 'Extracted Text (OCR)' : 'Generated Caption Description';
            consoleVerdict.innerHTML = `
                <div class="caption-box">
                    <span>${titleLabel}</span>
                    <p>${textValue}</p>
                </div>
            `;
        }
    }
}

// Bounding Box Drawing Helper
function drawBboxes(bboxes, labels) {
    bboxes.forEach((box, index) => {
        // Florence-2 bboxes are returned in pixels coordinates: [x1, y1, x2, y2] or [y1, x1, y2, x2]
        // Actually, AutoProcessor post_process_generation returns them normalized as [x1, y1, x2, y2] pixels!
        const [x1, y1, x2, y2] = box;
        const label = labels[index] || `Object #${index + 1}`;
        const color = ANNOTATION_COLORS[index % ANNOTATION_COLORS.length];
        
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, Math.round(imageCanvas.width / 400)); // Dynamic thickness based on resolution
        ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
        ctx.shadowBlur = 4;
        
        // Draw Rect
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur = 0; // Reset
        
        // Draw Text Tag background
        ctx.fillStyle = color;
        const fontHeight = Math.max(12, Math.round(imageCanvas.width / 60));
        ctx.font = `bold ${fontHeight}px var(--font-sans)`;
        
        const tagText = ` ${label} `;
        const textWidth = ctx.measureText(tagText).width;
        
        // Draw tag background box above or inside
        const tagY = y1 - fontHeight - 4 > 0 ? y1 - fontHeight - 4 : y1;
        ctx.fillRect(x1, tagY, textWidth, fontHeight + 4);
        
        // Draw text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tagText, x1, tagY + fontHeight);
    });
}

// Polygon / Segmentation Drawing Helper
function drawPolygons(polygons, labels) {
    polygons.forEach((polyList, index) => {
        const color = ANNOTATION_COLORS[index % ANNOTATION_COLORS.length];
        
        // Draw each polygon in the list (an object can have multiple polygons, e.g. disconnected regions)
        polyList.forEach(points => {
            // points is [x1, y1, x2, y2, ...]
            if (points.length < 4) return;
            
            ctx.beginPath();
            ctx.moveTo(points[0], points[1]);
            for (let i = 2; i < points.length; i += 2) {
                ctx.lineTo(points[i], points[i+1]);
            }
            ctx.closePath();
            
            // Stroke polygon
            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(2, Math.round(imageCanvas.width / 300));
            ctx.stroke();
            
            // Fill polygon translucently for a premium glow layer
            ctx.fillStyle = convertHexToRgba(color, 0.3);
            ctx.fill();
        });
    });
}

function convertHexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ==========================================
// LIVE WEBCAM & STREAMING INFERENCE SECTION
// ==========================================
let webcamStream = null;
let webcamActive = false;
let streamingInterval = null;
const btnWebcam = document.getElementById('btn-webcam');
const webcamVideo = document.getElementById('webcam-video');

btnWebcam.addEventListener('click', async () => {
    if (webcamActive) {
        stopWebcam();
    } else {
        await startWebcam();
    }
});

async function startWebcam() {
    try {
        uploadedFile = null; // Clear static uploaded file
        btnRun.setAttribute('disabled', 'true');
        
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        
        webcamVideo.srcObject = webcamStream;
        webcamVideo.style.display = 'none'; // Keep hidden, render on canvas
        
        // Hide upload prompts and show canvas
        document.querySelector('.drop-prompt').style.display = 'none';
        canvasContainer.style.display = 'flex';
        depthOverlay.style.display = 'none';
        
        webcamActive = true;
        btnWebcam.innerHTML = '<i class="fa-solid fa-square-stop"></i> Stop Live Feed';
        btnWebcam.style.color = 'var(--color-neon-pink)';
        
        webcamVideo.onloadedmetadata = () => {
            imageCanvas.width = webcamVideo.videoWidth || 640;
            imageCanvas.height = webcamVideo.videoHeight || 480;
            
            // Start the streaming frame loop
            startStreamingLoop();
        };
        
    } catch (err) {
        console.error("Webcam startup error: ", err);
        alert("Could not access webcam. Ensure permissions are granted.");
        stopWebcam();
    }
}

function stopWebcam() {
    webcamActive = false;
    btnWebcam.innerHTML = '<i class="fa-solid fa-video"></i> Start Live Feed';
    btnWebcam.style.color = '';
    
    if (streamingInterval) {
        clearInterval(streamingInterval);
        streamingInterval = null;
    }
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;
    webcamVideo.style.display = 'none';
    
    // Reset Viewport to initial dropzone prompt
    document.querySelector('.drop-prompt').style.display = 'flex';
    canvasContainer.style.display = 'none';
    depthOverlay.style.display = 'none';
    
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    btnRun.setAttribute('disabled', 'true');
}

function startStreamingLoop() {
    // Run frame inference every 1200ms to balance MPS latency & frame rate
    streamingInterval = setInterval(async () => {
        if (!webcamActive) return;
        
        // 1. Draw current video frame to canvas
        ctx.drawImage(webcamVideo, 0, 0, imageCanvas.width, imageCanvas.height);
        
        // 2. Capture canvas frame as base64 jpeg
        const frameBase64 = imageCanvas.toDataURL('image/jpeg', 0.8);
        
        // Prepare request body
        const payload = {
            image: frameBase64,
            modality: currentModality
        };
        
        if (currentModality === 'florence') {
            payload.task = florenceTaskSelect.value;
            if (florenceTaskSelect.value === '<REFERRING_EXPRESSION_SEGMENTATION>') {
                payload.text_input = florenceTextInput.value || 'the object';
            }
        } else if (currentModality === 'clip') {
            payload.candidates = clipCandidatesTextarea.value || 'an image';
        }
        
        statusDot.className = 'status-dot loading';
        
        try {
            const start = performance.now();
            const response = await fetch('/api/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) return;
            const data = await response.json();
            
            const latencyMs = Math.round((performance.now() - start));
            statLatency.textContent = `${latencyMs} ms`;
            
            // Draw background frame again
            ctx.drawImage(webcamVideo, 0, 0, imageCanvas.width, imageCanvas.height);
            
            // Render predictions on top
            renderStreamResult(data);
            
        } catch (e) {
            console.error("Frame stream error:", e);
        } finally {
            statusDot.className = 'status-dot online';
        }
    }, 1200);
}

function renderStreamResult(data) {
    // Write Raw JSON
    rawJsonOutput.textContent = JSON.stringify(data, null, 2);
    
    if (currentModality === 'depth') {
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
        };
        img.src = data.depth_image;
        
        consoleVerdict.innerHTML = `
            <div class="caption-box" style="padding: 10px 16px;">
                <span>Live Depth Mapping</span>
                <p>Depth statistics: Mean = ${data.stats.mean.toFixed(2)} [Min: ${data.stats.min.toFixed(2)}, Max: ${data.stats.max.toFixed(2)}]</p>
            </div>
        `;
        return;
    }
    
    if (currentModality === 'clip') {
        let rows = '';
        data.matches.forEach((match, index) => {
            const percent = (match.probability * 100).toFixed(1);
            rows += `
                <tr>
                    <td><strong>#${index + 1}</strong></td>
                    <td>${match.label}</td>
                    <td>
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
                            <span style="font-family: var(--font-mono); font-weight: 500;">${percent}%</span>
                            <div style="flex: 1; max-width: 150px;">
                                <div class="prob-bar-container">
                                    <div class="prob-bar" style="width: ${percent}%;"></div>
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        consoleVerdict.innerHTML = `
            <table class="similarity-table">
                <thead>
                    <tr>
                        <th width="80px">Rank</th>
                        <th>Tag/Candidate</th>
                        <th>Probability Match</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
        `;
        return;
    }
    
    if (currentModality === 'florence') {
        const task = florenceTaskSelect.value;
        const resultPayload = data.result[task];
        
        if (!resultPayload) return;
        
        if (task === '<OD>' || task === '<DENSE_REGION_CAPTION>' || task === '<OCR_WITH_REGION>') {
            let boxes = [];
            let labels = [];
            
            if (task === '<OD>' || task === '<DENSE_REGION_CAPTION>') {
                boxes = resultPayload.bboxes || [];
                labels = resultPayload.labels || [];
            } else if (task === '<OCR_WITH_REGION>') {
                boxes = resultPayload.quad_boxes || [];
                labels = resultPayload.labels || [];
            }
            
            drawBboxes(boxes, labels);
            
            let boxesHtml = '';
            boxes.forEach((box, i) => {
                const color = ANNOTATION_COLORS[i % ANNOTATION_COLORS.length];
                const label = labels[i] || `Region #${i + 1}`;
                boxesHtml += `
                    <div class="prediction-box" style="margin-bottom: 6px; padding: 8px 12px;">
                        <div class="pred-left">
                            <span class="pred-color-tag" style="color: ${color}; width: 10px; height: 10px;"></span>
                            <span class="pred-label" style="font-size: 13px;">${label}</span>
                        </div>
                        <span class="pred-coords" style="font-size: 10px;">[${box.join(', ')}]</span>
                    </div>
                `;
            });
            consoleVerdict.innerHTML = boxesHtml;
            
        } else if (task === '<REFERRING_EXPRESSION_SEGMENTATION>') {
            const polygons = resultPayload.polygons || [];
            const labels = resultPayload.labels || [];
            drawPolygons(polygons, labels);
            
            consoleVerdict.innerHTML = `
                <div class="caption-box" style="padding: 10px 16px;">
                    <span>Live Segment Match</span>
                    <p>Segmenting target: <strong>"${florenceTextInput.value}"</strong> (${polygons.length} masks)</p>
                </div>
            `;
            
        } else if (task === '<CAPTION>' || task === '<DETAILED_CAPTION>' || task === '<MORE_DETAILED_CAPTION>' || task === '<OCR>') {
            const textValue = typeof resultPayload === 'string' ? resultPayload : JSON.stringify(resultPayload);
            const titleLabel = task === '<OCR>' ? 'Extracted Text (OCR)' : 'Generated Caption';
            consoleVerdict.innerHTML = `
                <div class="caption-box" style="padding: 10px 16px;">
                    <span>${titleLabel}</span>
                    <p>${textValue}</p>
                </div>
            `;
        }
    }
}
