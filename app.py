import math
import time
import requests
from flask import Flask, render_template, request, jsonify # Import jsonify for JSON responses
import json
import os 

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main page with a form to input coordinates."""
    return render_template('index.html')

@app.route('/api/find_places', methods=['POST']) # Changed route to API endpoint
def find_places_api():
    """
    Processes the submitted coordinates, runs the pipeline, and returns results as JSON.
    Handles potential errors during coordinate parsing or pipeline execution.
    """
    coords_json_text = request.form['coords_json']
    # Extract lambda value from form data
    lambda_value = float(request.form.get('lambda', 0.7)) # Default to 0.7 if not provided or invalid

    origins = []
    
    if not coords_json_text:
        return jsonify({"error": "No valid origin coordinates provided."}), 400

    try:
        origins = json.loads(coords_json_text)
        if not isinstance(origins, list) or not all(isinstance(o, dict) and 'lat' in o and 'lng' in o for o in origins):
            raise ValueError("Invalid JSON format for coordinates.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing coordinates JSON: {e}")
        return jsonify({"error": f"Coordinate format error: {e}"}), 400

    if not origins:
        return jsonify({"error": "At least one origin coordinate must be selected."}), 400

    try:
        # Pass the lambda_value to the pipeline function
        result_data = pipeline(origins, lam=lambda_value) 
        # Ensure 'best_station' and 'cafes_top10' are properly handled if None
        if result_data.get("best_station") is None:
            # If no best station found, return a specific message
            return jsonify({"message": "Could not find a suitable meeting station based on the provided origins.", "data": result_data}), 200
        
        # Return the structured result data as JSON
        return jsonify(result_data), 200
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# ===== Configuration Values =====
# Your Google API key is directly embedded here as requested.
GOOGLE_API_KEY = "AIzaSyD1gcVpxIcq6YyezPIbXpos51yiDlH5LyQ"


LAMBDA = 0.7 # This default is now overridden by the value from the form
GRID_SPACING_DEG = 0.002
MARGIN_DEG = 0.01

# ===== Common Utilities =====
def haversine_km(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two geographical points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def weighted_score(values, lam=LAMBDA): # lam parameter will now come from the pipeline call
    """Calculates a weighted score: score = λ * max_value + (1 - λ) * avg_value."""
    if not values:
        return float("inf")
    return lam * max(values) + (1 - lam) * (sum(values)/len(values))

# ===== Step 1-3: Generate grid candidates within origins bbox and find best point by distance =====
def make_grid_candidates(origins, spacing_deg=GRID_SPACING_DEG, margin_deg=MARGIN_DEG):
    """
    Generates a grid of candidate points within the bounding box of the origins,
    with an added margin.
    """
    if not origins:
        return []

    lats = [o["lat"] for o in origins]
    lngs = [o["lng"] for o in origins]
    min_lat, max_lat = min(lats) - margin_deg, max(lats) + margin_deg
    min_lng, max_lng = min(lngs) - margin_deg, max(lngs) + margin_deg

    if min_lat > max_lat: min_lat, max_lat = max_lat, min_lat
    if min_lng > max_lng: min_lng, max_lng = max_lng, min_lng

    grid = []
    lat = min_lat
    while lat <= max_lat:
        lng = min_lng
        while lng <= max_lng:
            grid.append({"lat": lat, "lng": lng})
            lng += spacing_deg
        lat += spacing_deg
    return grid

def best_point_by_distance(origins, grid_pts, lam=LAMBDA): # lam parameter will now come from the pipeline call
    """
    Finds the best point from the grid based on the weighted distance score
    to all origin points.
    """
    if not grid_pts or not origins:
        return None

    best = None
    best_score = float("inf")
    for p in grid_pts:
        dists = [haversine_km(o["lat"], o["lng"], p["lat"], p["lng"]) for o in origins]
        s = weighted_score(dists, lam)
        if s < best_score:
            best_score = s
            best = {**p, "score": s, "max_km": max(dists), "avg_km": sum(dists)/len(dists)}
    return best

# ===== Step 4: Get transit stations within radius of the best point (Places Nearby) =====
def places_nearby(lat, lng, radius_m, type_name, api_key=GOOGLE_API_KEY):
    """
    Performs a Google Places Nearby Search to find places of a specified type
    around a given location. Handles pagination to collect multiple pages of results.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": type_name,
        "key": api_key,
    }
    results = []
    page_count = 0
    while True:
        try:
            r = requests.get(url, params=params).json()
        except requests.exceptions.RequestException as e:
            print(f"Error making Places API request: {e}")
            break

        if r.get("status") != "OK":
            print(f"Places API error: {r.get('status')} - {r.get('error_message')}")
            break

        results.extend(r.get("results", []))
        next_token = r.get("next_page_token")
        if not next_token or page_count >= 2:
            break
        page_count += 1
        time.sleep(2.0)
        params["pagetoken"] = next_token

    cleaned = []
    for it in results:
        loc = it.get("geometry", {}).get("location")
        if loc:
            cleaned.append({
                "name": it.get("name"),
                "lat": loc["lat"],
                "lng": loc["lng"],
                "place_id": it.get("place_id")
            })
    return cleaned

def get_transit_stations_within_radius_expand(best_lat, best_lng, api_key=GOOGLE_API_KEY,
                                             start_km=1, max_km=5, min_count=5):
    """
    Searches for subway and train stations around a given point,
    expanding the search radius until a minimum count of stations is found
    or maximum radius is reached. Removes duplicates by place_id.
    """
    radius_km = start_km
    stations = []

    while radius_km <= max_km:
        subways = places_nearby(best_lat, best_lng, radius_m=radius_km*1000,
                                type_name="subway_station", api_key=api_key)
        trains  = places_nearby(best_lat, best_lng, radius_m=radius_km*1000,
                                type_name="train_station", api_key=api_key)

        seen = set()
        merged = []
        for x in subways + trains:
            pid = x.get("place_id")
            if pid and pid not in seen:
                merged.append(x)
                seen.add(pid)

        stations = merged
        if len(stations) >= min_count:
            break

        radius_km += 1

    return stations, radius_km


# ===== Step 5: Select the best station based on weighted travel time (Distance Matrix) =====
def distance_matrix_times_minutes(origins, destinations, mode="transit", api_key=GOOGLE_API_KEY):
    """
    Uses Google Distance Matrix API to get travel times (in minutes)
    from multiple origins to multiple destinations.
    """
    if not origins or not destinations:
        return []

    origins_param = "|".join([f"{o['lat']},{o['lng']}" for o in origins])
    dests_param   = "|".join([f"{d['lat']},{d['lng']}" for d in destinations])

    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origins_param,
        "destinations": dests_param,
        "mode": mode,
        "departure_time": "now",
        "key": api_key
    }
    
    try:
        r = requests.get(url, params=params).json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error making Distance Matrix API request: {e}")

    if r.get("status") != "OK":
        error_msg = r.get("error_message", "Unknown error")
        raise RuntimeError(f"DistanceMatrix API error: {r.get('status')} - {error_msg}")

    times = []
    for row in r.get("rows", []):
        row_times = []
        for el in row.get("elements", []):
            if el.get("status") == "OK":
                sec = el["duration"]["value"]
                row_times.append(sec/60.0)
            else:
                row_times.append(None)
        times.append(row_times)
    return times

def best_station_by_time(origins, stations, lam=LAMBDA, mode="transit", api_key=GOOGLE_API_KEY): # lam parameter will now come from the pipeline call
    """
    Finds the best station from a list of candidates based on the weighted travel
    time from all origins to that station.
    """
    if not stations:
        return None

    try:
        times_mat = distance_matrix_times_minutes(origins, stations, mode=mode, api_key=api_key)
    except RuntimeError as e:
        print(f"Failed to get distance matrix times: {e}")
        return None

    best = None
    best_score = float("inf")

    for j, st in enumerate(stations):
        col_times = []
        all_routes_available = True
        for i in range(len(origins)):
            t = times_mat[i][j]
            if t is None:
                all_routes_available = False
                break
            col_times.append(t)

        if not all_routes_available:
            continue

        s = weighted_score(col_times, lam)
        if s < best_score:
            best_score = s
            best = {
                **st,
                "score": s,
                "max_min": max(col_times),
                "avg_min": sum(col_times)/len(col_times),
                "times": col_times
            }
    return best

# ===== Step 6: Get top 10 cafes within 1km of the selected station, sorted by distance =====
def top10_cafes_within_1km_sorted_by_distance(station, api_key=GOOGLE_API_KEY):
    """
    Finds up to 10 cafes within a 1km radius of the given station,
    sorted by their distance from the station.
    """
    if not station:
        return []

    cafes = places_nearby(station["lat"], station["lng"], radius_m=1000, type_name="cafe", api_key=api_key)
    for c in cafes:
        c["distance_km"] = haversine_km(station["lat"], station["lng"], c["lat"], c["lng"])
    cafes.sort(key=lambda x: x["distance_km"])
    return cafes[:10]

# ===== Main Pipeline Function =====
def pipeline(origins, lam=LAMBDA, grid_spacing_deg=GRID_SPACING_DEG, margin_deg=MARGIN_DEG,
             travel_mode="transit", api_key=GOOGLE_API_KEY): # lam is now a parameter
    """
    Orchestrates the entire process of finding the optimal meeting point,
    then the best nearby transit station, and finally top 10 cafes near that station.
    """
    best_point = best_point_by_distance(origins,
                                        make_grid_candidates(origins, spacing_deg=grid_spacing_deg, margin_deg=margin_deg),
                                        lam=lam) # Pass lam to best_point_by_distance
    if not best_point:
        print("Could not find a best geographical point.")
        return {"best_point": None, "candidate_station_count": 0, "best_station": None, "cafes_top10": []}

    stations, used_radius = get_transit_stations_within_radius_expand(
        best_point["lat"], best_point["lng"], api_key=api_key,
        start_km=1, max_km=5, min_count=5
    )

    best_station = best_station_by_time(origins, stations, lam=lam, mode=travel_mode, api_key=api_key) # Pass lam to best_station_by_time

    cafes_top10 = []
    if best_station:
        cafes_top10 = top10_cafes_within_1km_sorted_by_distance(best_station, api_key=api_key)
    else:
        print("No best station found, skipping cafe search.")

    return {
        "best_point": best_point,
        "candidate_station_count": len(stations),
        "best_station": best_station,
        "cafes_top10": cafes_top10
    }

if __name__ == "__main__":
    print(" * Flask app starting on http://0.0.0.0:5000")
    print(" * Access it from your computer at http://127.0.0.1:5000 or http://localhost:5000")
    print(" * To access from other devices on your LOCAL NETWORK (e.g., phone on same Wi-Fi):")
    print("    1. Find your computer's local IP address (e.g., run 'ipconfig' on Windows or 'ifconfig' on macOS/Linux).")
    print("    2. On your device, open a browser and go to http://YOUR_COMPUTERS_IP:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

