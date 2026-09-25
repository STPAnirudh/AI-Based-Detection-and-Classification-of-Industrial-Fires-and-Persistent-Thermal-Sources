// State variables
let map;
let allEvents = [];
let allFacilities = [];
let eventMarkersLayer;
let facilityMarkersLayer;
let proximityLineLayer;
let selectedEventId = null;

// Color mapping for prototype categories
const CATEGORY_COLORS = {
  "Potential Industrial Fire": "#ef4444",
  "Persistent Thermal Source": "#f97316",
  "Natural/Vegetation Fire": "#10b981",
  "Other/Uncertain": "#94a3b8"
};

// Priority badge classes
const PRIORITY_CLASSES = {
  "HIGH": "prio-high",
  "MEDIUM": "prio-medium",
  "LOW": "prio-low"
};

// Category badge classes
const CATEGORY_CLASSES = {
  "Potential Industrial Fire": "cat-industrial",
  "Persistent Thermal Source": "cat-persistent",
  "Natural/Vegetation Fire": "cat-vegetation",
  "Other/Uncertain": "cat-uncertain"
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadData();
  setupFilterListeners();
  setupSimulator();
});

function initMap() {
  // Center coordinates: Visakhapatnam Industrial Corridor
  const defaultCenter = [17.67, 83.20];
  const defaultZoom = 11;

  map = L.map("map", {
    center: defaultCenter,
    zoom: defaultZoom,
    zoomControl: true
  });

  // Base Map Layers
  const osmStandard = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  });

  const esriSatellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
  });

  const osmDark = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap'
  });

  // Default to OpenStreetMap standard layer (publicly accessible, watermark-free)
  osmStandard.addTo(map);

  const baseMaps = {
    "Street Map (OpenStreetMap)": osmStandard,
    "Satellite Imagery (Esri)": esriSatellite,
    "Dark Matter (Carto)": osmDark
  };

  eventMarkersLayer = L.layerGroup().addTo(map);
  facilityMarkersLayer = L.layerGroup().addTo(map);
  proximityLineLayer = L.layerGroup().addTo(map);

  const overlayMaps = {
    "Thermal Hotspots": eventMarkersLayer,
    "Industrial Facilities": facilityMarkersLayer
  };

  L.control.layers(baseMaps, overlayMaps, { position: "topright" }).addTo(map);
}

async function loadData() {
  try {
    // 1. Fetch Summary Statistics
    const statsRes = await fetch("/api/statistics");
    if (statsRes.ok) {
      const stats = await statsRes.json();
      updateStatisticsCards(stats);
    }

    // 2. Fetch Industrial Facilities
    const facRes = await fetch("/api/facilities");
    if (facRes.ok) {
      allFacilities = await facRes.json();
      renderFacilityMarkers(allFacilities);
    }

    // 3. Fetch Thermal Hotspot Events
    const evtRes = await fetch("/api/events");
    if (evtRes.ok) {
      allEvents = await evtRes.json();
      renderFilteredView();
      
      // Auto-select the highest risk event if present
      if (allEvents.length > 0) {
        const highRisk = allEvents.find(e => e.risk_priority === "HIGH") || allEvents[0];
        selectEvent(highRisk.event_id);
      }
    }
  } catch (err) {
    console.error("Error loading prototype data:", err);
  }
}

function updateStatisticsCards(stats) {
  document.getElementById("stat-total-events").textContent = stats.total_events || 0;
  document.getElementById("stat-ind-fires").textContent = stats.potential_industrial_fires || 0;
  document.getElementById("stat-persistent").textContent = stats.persistent_sources || 0;
  
  const vegCount = (stats.counts_by_class && stats.counts_by_class["Natural/Vegetation Fire"]) || 0;
  document.getElementById("stat-vegetation").textContent = vegCount;
  
  document.getElementById("stat-high-priority").textContent = stats.high_risk_count || 0;
}

function renderFacilityMarkers(facilities) {
  facilityMarkersLayer.clearLayers();

  facilities.forEach(fac => {
    // Custom blue marker for industrial facilities
    const facIcon = L.divIcon({
      className: "facility-marker-icon",
      html: `<div style="background-color: #0284c7; width: 22px; height: 22px; border-radius: 4px; display: flex; align-items: center; justify-content: center; border: 2px solid #ffffff; box-shadow: 0 0 6px rgba(0,0,0,0.6); font-size: 11px;">🏭</div>`,
      iconSize: [22, 22],
      iconAnchor: [11, 11]
    });

    const marker = L.marker([fac.latitude, fac.longitude], { icon: facIcon });
    
    marker.bindPopup(`
      <div style="font-family: sans-serif;">
        <div class="popup-title">${fac.name}</div>
        <div class="popup-row"><strong>Type:</strong> ${fac.facility_type}</div>
        <div class="popup-row"><strong>Coords:</strong> ${fac.latitude.toFixed(4)}, ${fac.longitude.toFixed(4)}</div>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">${fac.description || ""}</div>
      </div>
    `);

    facilityMarkersLayer.addLayer(marker);
  });
}

function renderFilteredView() {
  const classFilter = document.getElementById("filter-class").value;
  const riskFilter = document.getElementById("filter-risk").value;
  const persistFilter = document.getElementById("filter-persistence").value;

  const filtered = allEvents.filter(evt => {
    const matchClass = (classFilter === "ALL" || evt.predicted_class === classFilter);
    const matchRisk = (riskFilter === "ALL" || evt.risk_priority === riskFilter);
    const matchPersist = (persistFilter === "ALL" || evt.persistence_level === persistFilter);
    return matchClass && matchRisk && matchPersist;
  });

  renderEventMarkers(filtered);
  renderEventsTable(filtered);
  document.getElementById("table-count").textContent = filtered.length;
}

function renderEventMarkers(events) {
  eventMarkersLayer.clearLayers();

  events.forEach(evt => {
    const color = CATEGORY_COLORS[evt.predicted_class] || "#94a3b8";
    const isHigh = evt.risk_priority === "HIGH";

    const circleMarker = L.circleMarker([evt.latitude, evt.longitude], {
      radius: isHigh ? 10 : 7,
      color: isHigh ? "#ffffff" : color,
      weight: isHigh ? 2.5 : 1.5,
      fillColor: color,
      fillOpacity: 0.85
    });

    circleMarker.bindTooltip(`
      <strong>${evt.event_id}</strong><br/>
      ${evt.predicted_class}<br/>
      FRP: ${evt.frp} MW | Risk: ${evt.risk_score}
    `, { direction: "top", offset: [0, -8] });

    circleMarker.on("click", () => {
      selectEvent(evt.event_id);
    });

    eventMarkersLayer.addLayer(circleMarker);
  });
}

function renderEventsTable(events) {
  const tbody = document.getElementById("events-table-body");
  tbody.innerHTML = "";

  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 16px;">No thermal events match the current filter selection.</td></tr>`;
    return;
  }

  events.forEach(evt => {
    const tr = document.createElement("tr");
    tr.id = `row-${evt.event_id}`;
    if (evt.event_id === selectedEventId) {
      tr.classList.add("selected");
    }

    const prioClass = PRIORITY_CLASSES[evt.risk_priority] || "prio-low";
    const catClass = CATEGORY_CLASSES[evt.predicted_class] || "cat-uncertain";

    tr.innerHTML = `
      <td><strong>${evt.event_id}</strong></td>
      <td>${evt.acq_date} ${evt.acq_time_fmt || ""}</td>
      <td><span class="badge-category ${catClass}">${evt.predicted_class}</span></td>
      <td>${evt.ml_confidence}%</td>
      <td>${evt.nearest_facility_name}</td>
      <td>${evt.distance_to_facility_km.toFixed(2)}</td>
      <td>${evt.persistence_level} (${evt.occurrence_count})</td>
      <td><strong>${evt.risk_score}</strong></td>
      <td><span class="badge-priority ${prioClass}">${evt.risk_priority}</span></td>
    `;

    tr.addEventListener("click", () => {
      selectEvent(evt.event_id);
    });

    tbody.appendChild(tr);
  });
}

function selectEvent(eventId) {
  selectedEventId = eventId;
  const evt = allEvents.find(e => e.event_id === eventId);
  if (!evt) return;

  // Update table row selection styling
  document.querySelectorAll("table.events-table tr").forEach(r => r.classList.remove("selected"));
  const row = document.getElementById(`row-${eventId}`);
  if (row) {
    row.classList.add("selected");
    row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Draw Proximity Line to Nearest Facility
  proximityLineLayer.clearLayers();
  const fac = allFacilities.find(f => f.facility_id === evt.nearest_facility_id);
  if (fac) {
    const polyline = L.polyline([
      [evt.latitude, evt.longitude],
      [fac.latitude, fac.longitude]
    ], {
      color: "#38bdf8",
      weight: 2,
      dashArray: "5, 7",
      opacity: 0.8
    });
    
    polyline.bindTooltip(`Proximity: ${evt.distance_to_facility_km.toFixed(2)} km to ${fac.name}`, { sticky: true });
    proximityLineLayer.addLayer(polyline);
  }

  // Pan map smoothly to the event
  map.panTo([evt.latitude, evt.longitude], { animate: true });

  // Update Right Details Panel
  document.getElementById("empty-state-notice").style.display = "none";
  document.getElementById("event-details-content").style.display = "flex";

  document.getElementById("detail-title").textContent = evt.event_id;

  const catBadge = document.getElementById("detail-badge-category");
  catBadge.style.display = "inline-block";
  catBadge.textContent = evt.predicted_class;
  catBadge.className = `badge-category ${CATEGORY_CLASSES[evt.predicted_class] || "cat-uncertain"}`;

  const prioBadge = document.getElementById("detail-priority-badge");
  prioBadge.textContent = `${evt.risk_priority} PRIORITY`;
  prioBadge.className = `badge-priority ${PRIORITY_CLASSES[evt.risk_priority] || "prio-low"}`;

  document.getElementById("detail-risk-score").textContent = `${evt.risk_score} / 100`;
  document.getElementById("detail-risk-explanation").textContent = evt.risk_explanation || "Multi-factor scoring heuristic applied.";

  document.getElementById("detail-coords").textContent = `${evt.latitude.toFixed(4)}° N, ${evt.longitude.toFixed(4)}° E`;
  document.getElementById("detail-datetime").textContent = `${evt.acq_date} ${evt.acq_time_fmt || ""}`;
  document.getElementById("detail-daynight").textContent = `Day/Night: ${evt.daynight || (evt.daynight_flag === 1 ? 'D' : 'N')}`;

  document.getElementById("detail-confidence").textContent = `${evt.ml_confidence}%`;
  document.getElementById("detail-persistence").textContent = `${evt.persistence_level} Level`;
  document.getElementById("detail-occurrences").textContent = `${evt.occurrence_count} Historic Detections`;

  document.getElementById("detail-facility-name").textContent = evt.nearest_facility_name;
  document.getElementById("detail-facility-sub").textContent = `Distance: ${evt.distance_to_facility_km.toFixed(2)} km | Sector: ${evt.nearest_facility_type}`;

  document.getElementById("detail-frp").textContent = `${evt.frp} MW`;
  document.getElementById("detail-brightness").textContent = `${evt.brightness} K`;
  document.getElementById("detail-bright-t31").textContent = `T31: ${evt.bright_t31} K`;

  document.getElementById("detail-obs-window").textContent = `First: ${evt.first_observed_date} | Latest: ${evt.latest_observed_date} (${evt.days_span} days)`;
  document.getElementById("detail-cluster-id").textContent = `Cluster: ${evt.cluster_id || "CLUS_001"}`;
}

function setupFilterListeners() {
  document.getElementById("filter-class").addEventListener("change", renderFilteredView);
  document.getElementById("filter-risk").addEventListener("change", renderFilteredView);
  document.getElementById("filter-persistence").addEventListener("change", renderFilteredView);

  document.getElementById("btn-reset-filters").addEventListener("click", () => {
    document.getElementById("filter-class").value = "ALL";
    document.getElementById("filter-risk").value = "ALL";
    document.getElementById("filter-persistence").value = "ALL";
    renderFilteredView();
  });
}

function setupSimulator() {
  const btn = document.getElementById("btn-run-sim");
  btn.addEventListener("click", async () => {
    const brightness = parseFloat(document.getElementById("sim-brightness").value) || 350.0;
    const frp = parseFloat(document.getElementById("sim-frp").value) || 20.0;
    const dist = parseFloat(document.getElementById("sim-dist").value) || 1.0;
    const conf = parseFloat(document.getElementById("sim-conf").value) || 80.0;

    btn.textContent = "Classifying...";
    btn.disabled = true;

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brightness: brightness,
          bright_t31: brightness - 50.0,
          frp: frp,
          distance_to_facility_km: dist,
          daynight_flag: 1,
          confidence_score: conf,
          persistence_level: dist < 1.0 ? "Medium" : "Low"
        })
      });

      if (res.ok) {
        const data = await res.json();
        const resBox = document.getElementById("sim-result");
        resBox.style.display = "block";
        document.getElementById("sim-res-class").textContent = data.predicted_class;
        document.getElementById("sim-res-conf").textContent = `${data.ml_confidence_percent}%`;
        document.getElementById("sim-res-score").textContent = `${data.risk_score} / 100`;
        document.getElementById("sim-res-prio").textContent = data.risk_priority;
      }
    } catch (err) {
      console.error("Simulator error:", err);
    } finally {
      btn.textContent = "Run Live AI Classification";
      btn.disabled = false;
    }
  });
}
