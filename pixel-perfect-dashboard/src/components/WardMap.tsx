import { useMemo, useRef, useState, type MouseEvent } from "react";
import { MapContainer, TileLayer, GeoJSON as GeoJSONLayer, useMap } from "react-leaflet";
import type { Layer, GeoJSON as LeafletGeoJSON, LeafletMouseEvent } from "leaflet";
import { Crosshair, Layers3, ZoomIn, ZoomOut } from "lucide-react";
import "leaflet/dist/leaflet.css";

import type { WardFeature, WardGeoJSON } from "@/lib/api";

export type MapMode = "Risk" | "Persistence" | "Future Risk";

const BENGALURU_CENTER: [number, number] = [12.9716, 77.5946];

const UNKNOWN_COLOR = "#4b5563";

const LEVEL_COLORS: Record<string, string> = {
  Low: "#a3d977",
  Moderate: "#e8b84b",
  High: "#e05a3a",
  "Very High": "#c22a2a",
  Unknown: UNKNOWN_COLOR,
};

function probabilityColor(value: number | null): string {
  if (value === null || Number.isNaN(value)) return UNKNOWN_COLOR;
  if (value >= 0.7) return LEVEL_COLORS["High"] ?? UNKNOWN_COLOR;
  if (value >= 0.39) return LEVEL_COLORS["Moderate"] ?? UNKNOWN_COLOR;
  return LEVEL_COLORS["Low"] ?? UNKNOWN_COLOR;
}

function colorForWard(mode: MapMode, properties: WardFeature["properties"]): string {
  if (mode === "Risk") return LEVEL_COLORS[properties.priority_level] ?? UNKNOWN_COLOR;
  if (mode === "Persistence") return LEVEL_COLORS[properties.persistence_level] ?? UNKNOWN_COLOR;
  return probabilityColor(properties.future_heat_probability);
}

interface MapControlsProps {
  onLayersClick?: () => void;
}

function MapControls({ onLayersClick }: MapControlsProps) {
  const map = useMap();

  const stop = (event: MouseEvent) => event.stopPropagation();

  return (
    <div className="map-tools" onMouseDown={stop} onDoubleClick={stop}>
      <button type="button" aria-label="Zoom in" onClick={() => map.zoomIn()}>
        <ZoomIn />
      </button>
      <button type="button" aria-label="Zoom out" onClick={() => map.zoomOut()}>
        <ZoomOut />
      </button>
      <button
        type="button"
        aria-label="Center map"
        onClick={() => map.setView(BENGALURU_CENTER, 11)}
      >
        <Crosshair />
      </button>
      <button type="button" aria-label="Map layers" onClick={onLayersClick}>
        <Layers3 />
      </button>
    </div>
  );
}

interface WardTooltipState {
  x: number;
  y: number;
  properties: WardFeature["properties"];
}

interface WardMapProps {
  data: WardGeoJSON | undefined;
  mode: MapMode;
  onWardSelect?: (wardKey: string) => void;
}

export function WardMap({ data, mode, onWardSelect }: WardMapProps) {
  const [tooltip, setTooltip] = useState<WardTooltipState | null>(null);
  const geoJsonRef = useRef<LeafletGeoJSON | null>(null);

  const styleFn = useMemo(
    () => (feature?: WardFeature) => ({
      color: "rgba(255,255,255,0.35)",
      weight: 0.6,
      fillOpacity: 0.75,
      fillColor: feature ? colorForWard(mode, feature.properties) : UNKNOWN_COLOR,
    }),
    [mode],
  );

  // react-leaflet's GeoJSON layer only reads `data` once, at construction.
  // Remount it whenever the actual set of rendered wards changes (not just
  // the mode), so filtered views on the Explore page redraw correctly.
  const layerKey = useMemo(
    () => `${mode}:${data?.features.length ?? 0}:${data?.features.map((f) => f.properties.ward_key).join(",") ?? ""}`,
    [mode, data],
  );

  const onEachFeature = (feature: WardFeature, layer: Layer) => {
    layer.on({
      mouseover: (event: LeafletMouseEvent) => {
        (event.target as { setStyle: (style: object) => void }).setStyle({
          weight: 2,
          color: "#ffffff",
        });
        setTooltip({
          x: event.containerPoint.x,
          y: event.containerPoint.y,
          properties: feature.properties,
        });
      },
      mousemove: (event: LeafletMouseEvent) => {
        setTooltip((current) =>
          current ? { ...current, x: event.containerPoint.x, y: event.containerPoint.y } : current,
        );
      },
      mouseout: (event: LeafletMouseEvent) => {
        geoJsonRef.current?.resetStyle(event.target as never);
        setTooltip(null);
      },
      click: () => onWardSelect?.(feature.properties.ward_key),
    });
  };

  return (
    <div className="map-stage">
      <MapContainer
        center={BENGALURU_CENTER}
        zoom={11}
        scrollWheelZoom
        zoomControl={false}
        style={{ width: "100%", height: "100%", background: "var(--background)" }}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          className="dark-basemap-tiles"
        />
        {data && (
          <GeoJSONLayer
            key={layerKey}
            ref={geoJsonRef}
            data={data as never}
            style={styleFn as never}
            onEachFeature={onEachFeature as never}
          />
        )}
        <MapControls />
      </MapContainer>

      {tooltip && (
        <div
          className="map-tooltip"
          style={{ left: tooltip.x + 16, top: tooltip.y - 20, position: "absolute" }}
        >
          <strong>Ward: {tooltip.properties.ward_key}</strong>
          <span>Area: {tooltip.properties.ward_name}</span>
          <span>
            Risk Level:{" "}
            <b style={{ color: LEVEL_COLORS[tooltip.properties.priority_level] ?? UNKNOWN_COLOR }}>
              ● {tooltip.properties.priority_level}
            </b>
          </span>
          <span>
            LST:{" "}
            {tooltip.properties.mean_lst_celsius !== null
              ? `${tooltip.properties.mean_lst_celsius.toFixed(1)}°C`
              : "No data"}
          </span>
          <span>
            Future Risk:{" "}
            {tooltip.properties.future_heat_probability !== null
              ? `${(tooltip.properties.future_heat_probability * 100).toFixed(1)}%`
              : "No data"}
          </span>
          <em>Click ward for details →</em>
        </div>
      )}

      <div className="legend">
        <span>Lower Risk</span>
        <i />
        <span>Higher Risk</span>
      </div>
    </div>
  );
}
