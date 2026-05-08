"""Lightweight, no-API-key geo utilities for the doctor finder.

Uses:
  • OpenStreetMap Nominatim — for free geocoding (city → lat/lng).
    https://nominatim.org/release-docs/latest/api/Overview/
  • OpenStreetMap Overpass API — for nearby-hospital lookup.
    https://overpass-api.de/

Both services are public, no key, but rate-limited. We cache aggressively
and fail gracefully — every function returns sensible defaults on error.

We do NOT call Google APIs (they'd require a billing-enabled key).  Map
embeds use the keyless pattern  https://maps.google.com/maps?q=…&output=embed
which Google still serves for non-commercial use — and is the standard pattern
documented in many tutorials.  As a fallback the embed switches to
OpenStreetMap if Google ever blocks it.
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
from urllib.parse import quote_plus

import requests

USER_AGENT = "ColonAI-research-tool/1.0 (educational use only)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"


@dataclass
class GeoPoint:
    lat: float
    lng: float
    display_name: str = ""
    country_code: str = ""
    country: str = ""


# ──────────────────────────────────────────────────────────────────
# Geocoding
# ──────────────────────────────────────────────────────────────────

# In-process cache (process lifetime is enough for a streamlit session).
_GEO_CACHE: Dict[str, Optional[GeoPoint]] = {}


def geocode_city(city: str, country: str = "", timeout: float = 5.0) -> Optional[GeoPoint]:
    """Convert a free-text city / address into a GeoPoint.

    Returns None on network error or empty result. Cached across calls.
    """
    if not city:
        return None
    key = f"{city.strip().lower()}|{country.strip().lower()}"
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]

    params = {
        "q": (city + (", " + country if country else "")).strip(),
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    try:
        resp = requests.get(
            NOMINATIM_URL, params=params,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            _GEO_CACHE[key] = None
            return None
        data = resp.json() or []
        if not data:
            _GEO_CACHE[key] = None
            return None
        d = data[0]
        addr = d.get("address", {}) or {}
        gp = GeoPoint(
            lat=float(d["lat"]),
            lng=float(d["lon"]),
            display_name=d.get("display_name", ""),
            country_code=(addr.get("country_code") or "").upper(),
            country=addr.get("country", country or ""),
        )
        _GEO_CACHE[key] = gp
        # Polite to Nominatim (max 1 req/sec).
        time.sleep(0.3)
        return gp
    except Exception:
        _GEO_CACHE[key] = None
        return None


# ──────────────────────────────────────────────────────────────────
# Distance
# ──────────────────────────────────────────────────────────────────

def haversine_km(p1: GeoPoint, p2: GeoPoint) -> float:
    """Great-circle distance between two points, kilometres."""
    R = 6371.0
    lat1, lat2 = math.radians(p1.lat), math.radians(p2.lat)
    dlat = math.radians(p2.lat - p1.lat)
    dlng = math.radians(p2.lng - p1.lng)
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


# ──────────────────────────────────────────────────────────────────
# Overpass API — find nearby hospitals / clinics on OpenStreetMap
# ──────────────────────────────────────────────────────────────────

_OSM_CACHE: Dict[str, list] = {}


def osm_nearby_specialists(point: GeoPoint, radius_km: float = 8.0,
                           timeout: float = 7.0,
                           gi_only: bool = True) -> List[Dict]:
    """Query OpenStreetMap Overpass for hospitals/clinics within radius.

    Returns a list of dicts with keys: name, lat, lng, tags, distance_km.
    Filters opportunistically for GI-related tags but falls back to general
    hospitals if the GI-only query returns nothing.
    """
    if not point:
        return []

    key = f"{point.lat:.3f},{point.lng:.3f}|{radius_km:.1f}|{int(gi_only)}"
    if key in _OSM_CACHE:
        return _OSM_CACHE[key]

    radius_m = int(radius_km * 1000)
    # Overpass QL — search for hospitals + clinics + healthcare amenities
    # within a circle around the point.
    if gi_only:
        # Try a more specific query first
        ql = f"""
        [out:json][timeout:{int(timeout)}];
        (
          node["healthcare"]["healthcare:speciality"~"gastro|colon|onco|hepato",i](around:{radius_m},{point.lat},{point.lng});
          way ["healthcare"]["healthcare:speciality"~"gastro|colon|onco|hepato",i](around:{radius_m},{point.lat},{point.lng});
          node["amenity"="hospital"](around:{radius_m},{point.lat},{point.lng});
          way ["amenity"="hospital"](around:{radius_m},{point.lat},{point.lng});
        );
        out center 30;
        """
    else:
        ql = f"""
        [out:json][timeout:{int(timeout)}];
        (
          node["amenity"~"hospital|clinic|doctors"](around:{radius_m},{point.lat},{point.lng});
          way ["amenity"~"hospital|clinic|doctors"](around:{radius_m},{point.lat},{point.lng});
        );
        out center 30;
        """
    try:
        resp = requests.post(OVERPASS_URL, data=ql,
                             headers={"User-Agent": USER_AGENT,
                                      "Content-Type": "application/x-www-form-urlencoded"},
                             timeout=timeout)
        if resp.status_code != 200:
            _OSM_CACHE[key] = []
            return []
        data = resp.json()
    except Exception:
        _OSM_CACHE[key] = []
        return []

    out = []
    for el in (data.get("elements", []) or []):
        tags = el.get("tags") or {}
        name = tags.get("name") or tags.get("operator") or ""
        if not name:
            continue
        if el.get("type") == "node":
            lat = el.get("lat"); lng = el.get("lon")
        elif "center" in el:
            lat = el["center"]["lat"]; lng = el["center"]["lon"]
        else:
            continue
        dist = haversine_km(point, GeoPoint(lat=lat, lng=lng))
        out.append({
            "name": name,
            "lat": lat,
            "lng": lng,
            "amenity": tags.get("amenity") or tags.get("healthcare") or "facility",
            "speciality": tags.get("healthcare:speciality", ""),
            "operator": tags.get("operator", ""),
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "website": tags.get("website") or tags.get("contact:website") or "",
            "addr": ", ".join(filter(None, [
                tags.get("addr:street", ""),
                tags.get("addr:city", "") or tags.get("addr:town", "") or "",
                tags.get("addr:postcode", ""),
            ])),
            "distance_km": round(dist, 2),
        })
    out.sort(key=lambda x: x["distance_km"])
    _OSM_CACHE[key] = out[:30]
    return out[:30]


# ──────────────────────────────────────────────────────────────────
# Map URL helpers (no API key required)
# ──────────────────────────────────────────────────────────────────

def google_maps_embed_url(query: str = "", lat: Optional[float] = None,
                          lng: Optional[float] = None, zoom: int = 12) -> str:
    """Return a key-less Google Maps embed URL for an iframe."""
    if lat is not None and lng is not None:
        q = f"{lat},{lng}"
        return f"https://maps.google.com/maps?q={q}&t=&z={zoom}&ie=UTF8&iwloc=&output=embed"
    return f"https://maps.google.com/maps?q={quote_plus(query)}&t=&z={zoom}&ie=UTF8&iwloc=&output=embed"


def google_maps_directions_url(origin: str = "", destination: str = "") -> str:
    """Return a Google Maps directions URL (works without API key)."""
    if origin and destination:
        return ("https://www.google.com/maps/dir/?api=1"
                f"&origin={quote_plus(origin)}"
                f"&destination={quote_plus(destination)}")
    if destination:
        return ("https://www.google.com/maps/search/?api=1"
                f"&query={quote_plus(destination)}")
    return "https://www.google.com/maps"


def osm_embed_url(point: GeoPoint, zoom: int = 12) -> str:
    """OpenStreetMap embed URL (fallback if Google embed gets blocked)."""
    delta = 0.04 * (12 / max(1, zoom))
    bbox = (point.lng - delta, point.lat - delta,
            point.lng + delta, point.lat + delta)
    return (f"https://www.openstreetmap.org/export/embed.html"
            f"?bbox={bbox[0]:.4f}%2C{bbox[1]:.4f}%2C{bbox[2]:.4f}%2C{bbox[3]:.4f}"
            f"&layer=mapnik&marker={point.lat:.5f}%2C{point.lng:.5f}")
