document.addEventListener("DOMContentLoaded", () => {
    const uploadInput = document.getElementById("upload");
    const resultsDiv = document.getElementById("results");
    const imageElement = document.getElementById("image");
    const canvas = document.getElementById("canvas");
    const filterButtons = document.getElementById("filter-buttons");
    const saveDetectionButton = document.getElementById("save-detection");
    const locationSelect = document.getElementById("location-select");
    const locationCard = document.getElementById("location-card");
    const mapDiv = document.getElementById("map");
    const detectedLocation = document.getElementById("detected-location");
    const exportButton = document.getElementById("export-geojson");
    const opencvInfo = document.getElementById("opencvinfo");
    const onnxInfo = document.getElementById("onnxinfo");

    // Netlify serverless function URL for saving coordinates
    const ADD_COORDINATES_URL = 'https://mongo-atlas-serverless.netlify.app/.netlify/functions/add-coordinates';

    // Map variables
    let map = null;
    let marker = null;
    let detections = []; // Store detections for GeoJSON export
    let currentCoords = null;

    if (!uploadInput || !resultsDiv || !imageElement || !canvas || !filterButtons || !saveDetectionButton || !locationSelect || !locationCard || !mapDiv || !detectedLocation || !exportButton || !opencvInfo || !onnxInfo) {
        console.error("One or more DOM elements not found");
        resultsDiv.innerHTML = "<p>Error: Page elements not found</p>";
        return;
    }

    // Initialize map function
    function initializeMap() {
        if (map) return; // Already initialized
        map = L.map(mapDiv).setView([-29.0, 24.0], 6);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }).addTo(map);

        map.on('click', function (e) {
            if (marker) {
                marker.setLatLng(e.latlng);
            } else {
                marker = L.marker(e.latlng, { draggable: true }).addTo(map);
                marker.on('dragend', function (e) {
                    const pos = marker.getLatLng();
                    detectedLocation.textContent = `Location: ${pos.lat.toFixed(6)}, ${pos.lng.toFixed(6)}`;
                    currentCoords = { latitude: pos.lat, longitude: pos.lng };
                });
            }
            detectedLocation.textContent = `Location: ${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
            detectedLocation.style.display = 'block';
            currentCoords = { latitude: e.latlng.lat, longitude: e.latlng.lng };
        });
    }

    // Convert GPS coordinates from EXIF format to decimal degrees
    function convertDMSToDD(degrees, minutes, seconds, direction) {
        let dd = degrees + minutes / 60 + seconds / 3600;
        if (direction === 'S' || direction === 'W') {
            dd = dd * -1;
        }
        return dd;
    }

    // Extract GPS data from EXIF
    function extractGPSFromEXIF(file) {
        return new Promise((resolve, reject) => {
            if (typeof EXIF === 'undefined') {
                console.warn('EXIF.js not loaded');
                resolve(null);
                return;
            }

            EXIF.getData(file, function () {
                const lat = EXIF.getTag(this, "GPSLatitude");
                const latRef = EXIF.getTag(this, "GPSLatitudeRef");
                const lon = EXIF.getTag(this, "GPSLongitude");
                const lonRef = EXIF.getTag(this, "GPSLongitudeRef");

                console.log('EXIF GPS Data:', { lat, latRef, lon, lonRef });

                if (lat && lon && latRef && lonRef) {
                    const latitude = convertDMSToDD(lat[0], lat[1], lat[2], latRef);
                    const longitude = convertDMSToDD(lon[0], lon[1], lon[2], lonRef);
                    console.log('Converted coordinates:', { latitude, longitude });
                    resolve({ latitude, longitude });
                } else {
                    console.log('No GPS data found in EXIF');
                    resolve(null);
                }
            });
        });
    }

    // Set marker on map with coordinates
    function setMarkerAtCoordinates(latitude, longitude, source) {
        if (!map) {
            initializeMap();
            setTimeout(() => setMarkerAtCoordinates(latitude, longitude, source), 200);
            return;
        }

        map.setView([latitude, longitude], 15);
        if (marker) {
            marker.setLatLng([latitude, longitude]);
        } else {
            marker = L.marker([latitude, longitude], { draggable: true }).addTo(map);
            marker.on('dragend', function (e) {
                const pos = marker.getLatLng();
                detectedLocation.textContent = `Location: ${pos.lat.toFixed(6)}, ${pos.lng.toFixed(6)}`;
                currentCoords = { latitude: pos.lat, longitude: pos.lng };
            });
        }
        detectedLocation.textContent = `${source}: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}`;
        detectedLocation.style.display = 'block';
        detectedLocation.style.color = 'var(--accent)';
        currentCoords = { latitude, longitude };
    }

    // Province centroids for fallback
    const PROVINCE_CENTROIDS = {
        "Eastern Cape": [-32.2968, 26.4194],
        "Free State": [-28.4547, 26.7968],
        "Gauteng": [-26.2708, 28.1123],
        "KwaZulu-Natal": [-29.6035, 30.3794],
        "Limpopo": [-23.4013, 29.4179],
        "Mpumalanga": [-25.5653, 30.5279],
        "Northern Cape": [-29.0467, 21.8569],
        "North West": [-26.6639, 25.2838],
        "Western Cape": [-33.2278, 19.1492]
    };

    // Handle province selection
    locationSelect.addEventListener("change", () => {
        const province = locationSelect.value;
        if (province && map) {
            const coords = PROVINCE_CENTROIDS[province];
            if (coords) {
                map.setView([coords[0], coords[1]], 10);
                if (marker) {
                    marker.setLatLng([coords[0], coords[1]]);
                } else {
                    marker = L.marker([coords[0], coords[1]], { draggable: true }).addTo(map);
                    marker.on('dragend', function (e) {
                        const pos = marker.getLatLng();
                        detectedLocation.textContent = `Location: ${pos.lat.toFixed(6)}, ${pos.lng.toFixed(6)}`;
                        currentCoords = { latitude: pos.lat, longitude: pos.lng };
                    });
                }
                detectedLocation.textContent = `${province}: ${coords[0].toFixed(6)}, ${coords[1].toFixed(6)}`;
                detectedLocation.style.display = 'block';
                currentCoords = { latitude: coords[0], longitude: coords[1] };
            }
        }
    });

    // Softmax helper
    function softmax(arr) {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }

    // Preprocess and predict
    async function predictImage(file) {
        try {
            // Load ONNX model
            const session = await ort.InferenceSession.create('../models/model.onnx');

            // Preprocess image with OpenCV.js
            const img = await createImageBitmap(file);
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 224, 224);
            const mat = cv.imread(canvas);

            // Convert to tensor (RGB, normalized)
            const data = new Float32Array(3 * 224 * 224);
            const means = [0.485, 0.456, 0.406];
            const stds = [0.229, 0.224, 0.225];
            const imgData = ctx.getImageData(0, 0, 224, 224).data;
            for (let i = 0; i < 224 * 224; i++) {
                data[i] = (imgData[i * 4] / 255 - means[0]) / stds[0]; // R
                data[i + 224 * 224] = (imgData[i * 4 + 1] / 255 - means[1]) / stds[1]; // G
                data[i + 2 * 224 * 224] = (imgData[i * 4 + 2] / 255 - means[2]) / stds[2]; // B
            }
            const tensor = new ort.Tensor('float32', data, [1, 3, 224, 224]);

            // Run inference
            const outputs = await session.run({ input: tensor });
            const logits = outputs.output.data;
            const probabilities = softmax(logits);
            const predicted = probabilities[0] > probabilities[1] ? 0 : 1;
            const classNames = ['Pyracantha', 'Not Pyracantha'];

            // Display results
            resultsDiv.innerHTML = '';
            if (probabilities[1] * 100 >= 90) {
                resultsDiv.innerHTML += `
                    <p>Pyracantha Not Detected with confidence: ${(probabilities[1] * 100).toFixed(2)}%</p>
                `;
            } else {
                resultsDiv.innerHTML += `
                    <p>Pyracantha Detected with confidence: ${(probabilities[0] * 100).toFixed(2)}%</p>
                `;
            }

            onnxInfo.innerHTML = `<p>ONNX Runtime Web ${ort.version}</p>`;

            // Show or hide location card based on Pyracantha probability
            locationCard.style.display = probabilities[0] * 100 >= 90 ? 'block' : 'none';

            return classNames[predicted];
        } catch (e) {
            console.error('Prediction error:', e);
            resultsDiv.innerHTML = `<p class="error">Error: ${e.message}</p>`;
            locationCard.style.display = 'none';
            return null;
        }
    }

    // Handle file upload
    uploadInput.addEventListener("change", async () => {
        const file = uploadInput.files[0];
        if (!file) {
            console.warn("No file selected");
            resultsDiv.innerHTML = "<p>No image selected</p>";
            return;
        }

        console.log("File selected:", file.name, file.type);

        // Display image
        try {
            const imageUrl = URL.createObjectURL(file);
            imageElement.src = imageUrl;
            imageElement.style.display = "block";
            filterButtons.style.display = "flex";
            console.log("Image URL set:", imageUrl);

            imageElement.onload = () => {
                console.log("Image loaded successfully");
                URL.revokeObjectURL(imageUrl);
            };
            imageElement.onerror = () => {
                console.error("Failed to load image");
                resultsDiv.innerHTML = "<p>Error loading image</p>";
            };
        } catch (error) {
            console.error("Error setting image source:", error);
            resultsDiv.innerHTML = "<p>Error displaying image</p>";
        }

        // Extract EXIF GPS data and predict
        resultsDiv.innerHTML = "<p>Predicting…</p>";
        let gpsCoords = null;
        try {
            gpsCoords = await extractGPSFromEXIF(file);
            const prediction = await predictImage(file);
            if (locationCard.style.display === 'block') {
                initializeMap();
            }

            if (gpsCoords && locationCard.style.display === 'block') {
                setMarkerAtCoordinates(gpsCoords.latitude, gpsCoords.longitude, 'EXIF GPS Data');
            } else if (navigator.geolocation && locationCard.style.display === 'block') {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        setMarkerAtCoordinates(
                            position.coords.latitude,
                            position.coords.longitude,
                            'Browser Geolocation'
                        );
                    },
                    (error) => {
                        console.log('Geolocation error:', error.message);
                        detectedLocation.textContent = 'No location data available. Click on map or select province.';
                        detectedLocation.style.display = 'block';
                        detectedLocation.style.color = 'var(--muted)';
                    }
                );
            } else {
                detectedLocation.textContent = 'No location data available. Click on map or select province.';
                detectedLocation.style.display = 'block';
                detectedLocation.style.color = 'var(--muted)';
            }
        } catch (error) {
            console.error('Error processing image:', error);
            resultsDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    });

    // Save detection to MongoDB via Netlify function
    saveDetectionButton.addEventListener("click", async () => {
        if (!currentCoords) {
            alert('No location detected or selected');
            return;
        }
        const province = locationSelect.value || 'Unknown';
        const data = {
            latitude: currentCoords.latitude,
            longitude: currentCoords.longitude,
            label: resultsDiv.children[0]?.textContent.split(': ')[1]?.split(' with')[0] || 'Pyracantha',
            province,
            createdAt: new Date().toISOString()
        };

        try {
            const response = await fetch(ADD_COORDINATES_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            detections.push({
                ...data,
                _id: result.id
            });
            alert('Detection saved successfully!');
            resultsDiv.innerHTML += `<p class="success">✓ Detection saved</p>`;
        } catch (e) {
            console.error('Save error:', e);
            alert('Failed to save detection');
            resultsDiv.innerHTML += `<p class="error">Error saving detection: ${e.message}</p>`;
        }
    });

    // Export GeoJSON
    exportButton.addEventListener("click", () => {
        if (detections.length === 0 && !marker) {
            alert("No detections to export");
            return;
        }

        const geojson = {
            type: 'FeatureCollection',
            features: detections.map(d => ({
                type: 'Feature',
                geometry: { type: 'Point', coordinates: [d.longitude, d.latitude] },
                properties: {
                    id: d._id?.toString() || null,
                    label: d.label,
                    province: d.province,
                    createdAt: d.createdAt
                }
            }))
        };

        if (marker && currentCoords && !detections.some(d => d.latitude === currentCoords.latitude && d.longitude === currentCoords.longitude)) {
            geojson.features.push({
                type: 'Feature',
                geometry: { type: 'Point', coordinates: [currentCoords.longitude, currentCoords.latitude] },
                properties: {
                    label: resultsDiv.children[0]?.textContent.split(': ')[1]?.split(' with')[0] || 'Pyracantha',
                    province: locationSelect.value || 'Unknown',
                    createdAt: new Date().toISOString()
                }
            });
        }

        const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pyracantha_detections_${Date.now()}.geojson`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // Wait for OpenCV.js to load
    cv.onRuntimeInitialized = () => {
        opencvInfo.innerHTML = '<p>OpenCV.js loaded</p>';
    };

    // Filter buttons
    filterButtons.querySelectorAll('button').forEach(button => {
        button.addEventListener('click', (e) => {
            const filter = e.target.dataset.filter;
            canvas.width = imageElement.width;
            canvas.height = imageElement.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(imageElement, 0, 0);

            let mat = cv.imread(canvas);
            switch (filter) {
                case 'gray':
                    cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY);
                    break;
                case 'blur':
                    cv.GaussianBlur(mat, mat, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
                    break;
                case 'canny':
                    cv.cvtColor(mat, mat, cv.COLOR_RGBA2GRAY);
                    cv.Canny(mat, mat, 100, 200, 3, false);
                    break;
                case 'original':
                    break;
            }
            cv.imshow('canvas', mat);
            mat.delete();
            canvas.style.display = 'block';
            imageElement.style.display = 'none';
        });
    });
});