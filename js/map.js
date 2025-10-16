document.addEventListener("DOMContentLoaded", async () => {
    // Use PROVINCE_CENTROIDS from app.js
    const PROVINCE_CENTROIDS = window.PROVINCE_CENTROIDS;
    if (!PROVINCE_CENTROIDS) {
        console.error("PROVINCE_CENTROIDS not defined");
        return;
    }

    const mapEl = document.getElementById('map');
    if (!mapEl) {
        console.error("Map element not found");
        return;
    }

    // Initialize Leaflet map
    const map = L.map('map').setView([-30.5595, 22.9375], 6); // South Africa

    // OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);

    // Custom icon
    let icon;
    try {
        icon = L.icon({
            iconUrl: '../icons/plant.png', // Adjusted path; ensure file exists in /public/icons/
            iconSize: [32, 32],
            iconAnchor: [16, 32],
            popupAnchor: [0, -32]
        });
    } catch (e) {
        console.warn('Custom icon not found, using default Leaflet marker:', e);
        icon = new L.Icon.Default();
    }

    // Fetch coordinates from Netlify serverless function
    const GET_COORDINATES_URL = 'https://mongo-atlas-serverless.netlify.app/.netlify/functions/get-coordinates';

    try {
        const response = await fetch(GET_COORDINATES_URL, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) throw new Error(`HTTP error ${response.status}: ${response.statusText}`);
        const geojson = await response.json();

        if (!geojson || !geojson.features || geojson.features.length === 0) {
            mapEl.innerHTML = '<p style="color:blue; text-align:center;">No detections available. Upload an image to add coordinates.</p>';
            return;
        }

        const geoJsonLayer = L.geoJSON(geojson, {
            pointToLayer: (feature, latlng) => L.marker(latlng, { icon }),
            onEachFeature: (feature, layer) => {
                const p = feature.properties;
                layer.bindPopup(`
                    <b>${p.label || 'Narrowleaf Firethorn'}</b><br>
                    ID: ${p.id || 'N/A'}<br>
                    Latitude: ${feature.geometry.coordinates[1].toFixed(4)}<br>
                    Longitude: ${feature.geometry.coordinates[0].toFixed(4)}<br>
                    Province: ${p.province || 'Unknown'}<br>
                    Created: ${p.createdAt || 'N/A'}
                `);
            }
        }).addTo(map);

        const bounds = geoJsonLayer.getBounds();
        if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [50, 50] });
        } else {
            console.warn('Invalid bounds; unable to fit map to markers');
        }

    } catch (error) {
        console.error('Error fetching coordinates:', error);
        mapEl.innerHTML = `<p style="color:red; text-align:center;">Error loading coordinates: ${error.message}</p>`;
    }

    // Province jump control
    const select = L.DomUtil.create('select', 'leaflet-bar');
    select.style.padding = '6px';
    select.style.borderRadius = '4px';
    select.style.margin = '6px';

    const defaultOpt = document.createElement('option');
    defaultOpt.textContent = 'Jump to Province…';
    defaultOpt.value = '';
    select.appendChild(defaultOpt);

    Object.keys(PROVINCE_CENTROIDS).forEach(prov => {
        const option = document.createElement('option');
        option.textContent = prov;
        option.value = prov;
        select.appendChild(option);
    });

    select.onchange = () => {
        const prov = select.value;
        if (!prov) return;
        const coords = PROVINCE_CENTROIDS[prov];
        if (coords) map.setView(coords, 8);
    };

    const control = L.control({ position: 'topleft' });
    control.onAdd = () => {
        const container = L.DomUtil.create('div');
        container.appendChild(select);
        return container;
    };
    control.addTo(map);
});