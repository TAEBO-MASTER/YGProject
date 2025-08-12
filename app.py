import math
import time
import requests
from flask import Flask, render_template, request, jsonify
import json
import os 
from dotenv import load_dotenv

load_dotenv()
SERVER_KEY = os.getenv("GOOGLE_SERVER_API_KEY")
if not SERVER_KEY:
    raise RuntimeError("환경변수 GOOGLE_SERVER_API_KEY가 없습니다.")

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main page with a form to input coordinates."""
    return render_template('index.html')

@app.route("/route")
def route_detail():
    # 쿼리스트링으로 origin, dest, pref, depart를 받으므로 서버 로직 불필요
    return render_template("route_detail.html")

@app.route('/api/find_places', methods=['POST'])
def find_places_api():
    """
    Processes the submitted coordinates, runs the pipeline, and returns results as JSON.
    Handles potential errors during coordinate parsing or pipeline execution.
    """
    transit_pref = request.form.get("transit_pref", "FEWER_TRANSFERS")
    pref_map = {"LESS_WALKING": "less_walking", "FEWER_TRANSFERS": "fewer_transfers"}
    transit_pref_api = pref_map.get(transit_pref, "fewer_transfers")

    coords_json_text = request.form['coords_json']
    lambda_value = float(request.form.get('lambda', 0.7)) 

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
        result_data = pipeline(origins, lam=lambda_value, transit_pref_api=transit_pref_api)
        
        if result_data.get("result_type") == "error":
            return jsonify({"error": result_data.get("message", "An unknown error occurred.")}), 500

        return jsonify(result_data), 200
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# Default lambda is 0.7, but overridden by user input from slider
LAMBDA = 0.7 
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

def weighted_score(values, lam): # lam parameter is passed from pipeline/best_point/best_station_by_time
    """
    Calculates a weighted score: score = (1 - λ) * max_value + λ * avg_value.
    - If lambda is 0 (Equality): score = 1 * max_value + 0 * avg_value = max_value (minimize max distance/time)
    - If lambda is 1 (Efficiency): score = 0 * max_value + 1 * avg_value = avg_value (minimize average distance/time)
    """
    if not values:
        return float("inf")
    # Reversed weighting as requested: (1 - lam) for max_value (equality), lam for avg_value (efficiency)
    return (1 - lam) * max(values) + lam * (sum(values)/len(values))

# ===== Step 1-3: Generate grid candidates within origins bbox and find best point by distance =====
# ===== Step 1-3: Generate grid candidates with a fixed count (e.g., ~10k) =====
def make_grid_candidates_fixed_count(
    origins,
    target_points=10000,         # ≈ 100 x 100
    margin_deg=MARGIN_DEG,
    min_per_side=50,             # 너무 거칠지 않도록 하한
    max_per_side=250             # 과도한 계산 방지 상한
):
    """
    Bounding box 내부에 '점의 개수'를 고정해 균등 분포 시드 포인트를 생성합니다.
    bbox 종횡비를 유지하도록 행/열 수를 계산하고, 각 셀의 중심에 점을 둡니다.
    """
    if not origins:
        return []

    lats = [o["lat"] for o in origins]
    lngs = [o["lng"] for o in origins]
    min_lat, max_lat = min(lats) - margin_deg, max(lats) + margin_deg
    min_lng, max_lng = min(lngs) - margin_deg, max(lngs) + margin_deg

    height = max(max_lat - min_lat, 1e-9)
    width  = max(max_lng - min_lng, 1e-9)

    # bbox 종횡비(행/열 비율)에 맞춰 행/열 결정
    aspect = height / width
    n_rows = int(round(math.sqrt(target_points * aspect)))
    n_rows = max(2, min(max_per_side, max(min_per_side, n_rows)))
    n_cols = max(2, min(max_per_side, max(min_per_side, int(round(target_points / n_rows)))))

    # 셀 중심점으로 균등 샘플링 (경계 포함 대신 중심 사용)
    lat_step = height / n_rows
    lng_step = width  / n_cols

    grid = []
    for r in range(n_rows):
        lat = min_lat + (r + 0.5) * lat_step
        for c in range(n_cols):
            lng = min_lng + (c + 0.5) * lng_step
            grid.append({"lat": lat, "lng": lng})

    return grid

def best_point_by_distance(origins, grid_pts, lam): # lam parameter is passed from pipeline
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
# Modified places_nearby to accept keyword
def places_nearby(lat, lng, radius_m, type_name=None, keyword=None, api_key=SERVER_KEY):
    """
    Performs a Google Places Nearby Search to find places.
    Can specify type, keyword, or both.
    Handles pagination to collect multiple pages of results.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "key": api_key,
        "language": "ko"
    }
    if type_name:
        params["type"] = type_name
    if keyword:
        params["keyword"] = keyword
    
    # Google Places API requires either type or keyword for Nearby Search
    if not type_name and not keyword:
        # Fallback to a very general type if neither specified, to avoid API error
        params["type"] = "point_of_interest" # Or "establishment"

    results = []
    page_count = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=(5,15)).json()
        except requests.exceptions.RequestException as e:
            print(f"Error making Places API request: {e}")
            break

        if r.get("status") != "OK":
            print(f"Places API error for type {type_name} / keyword {keyword}: {r.get('status')} - {r.get('error_message')}")
            break

        results.extend(r.get("results", []))
        next_token = r.get("next_page_token")
        if not next_token or page_count >= 2: # Max 3 pages (initial + 2 next_page_tokens)
            break
        page_count += 1
        time.sleep(2.0) # Wait for next_page_token to become active (Google API requirement)
        params["pagetoken"] = next_token

    cleaned = []
    for it in results:
        loc = it.get("geometry", {}).get("location")
        if loc:
            cleaned.append({
                "name": it.get("name"),
                "lat": loc["lat"],
                "lng": loc["lng"],
                "place_id": it.get("place_id"),
                "types": it.get("types", []) # Include types for front-end differentiation
            })
    return cleaned

def get_transit_stations_within_radius_expand(best_lat, best_lng, api_key=SERVER_KEY,
                                             start_km=1, max_km=10, min_count=5): # Max radius now 10km
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

def get_best_bus_terminal(center_lat, center_lng, api_key=SERVER_KEY, start_km=1, max_km=5):
    """
    Searches for bus terminals around a given point, expanding the search radius.
    Returns the closest bus terminal found.
    """
    radius_km = start_km
    best_terminal = None
    min_dist = float('inf')

    while radius_km <= max_km:
        bus_stations = places_nearby(center_lat, center_lng, radius_m=radius_km*1000,
                                     type_name="bus_station", api_key=api_key)
        
        for bs in bus_stations:
            dist = haversine_km(center_lat, center_lng, bs["lat"], bs["lng"])
            if dist < min_dist:
                min_dist = dist
                best_terminal = {**bs, "distance_km": dist} # Add distance_km to the dict
        
        if best_terminal: # If any terminal found in current radius, return the closest one
            return best_terminal, radius_km
        
        radius_km += 1
    
    return None, 0 # No bus terminal found

# Modified get_nearby_places_sorted_by_distance to accept search_configs
def get_nearby_places_sorted_by_distance(center_lat, center_lng, search_configs, radius_m=1000, limit=10, api_key=SERVER_KEY):
    """
    Finds places based on a list of search configurations (type and/or keyword),
    within a radius of the given center, sorted by their distance from the center.
    Each config in search_configs should be a dict like {"type": "cafe"} or {"type": "point_of_interest", "keyword": "스터디"}.
    """
    if not search_configs:
        return []

    all_places_raw = []
    seen_place_ids = set()

    for config in search_configs:
        type_to_search = config.get("type")
        keyword_to_search = config.get("keyword")
        
        places_from_config = places_nearby(center_lat, center_lng, radius_m=radius_m, 
                                           type_name=type_to_search, keyword=keyword_to_search, api_key=api_key)
        for p in places_from_config:
            pid = p.get("place_id")
            if pid and pid not in seen_place_ids:
                all_places_raw.append(p)
                seen_place_ids.add(pid)
    
    cleaned_places = []
    for p in all_places_raw:
        if 'types' not in p:
            p['types'] = [] 
        
        p["distance_km"] = haversine_km(center_lat, center_lng, p["lat"], p["lng"])
        cleaned_places.append(p)
    
    cleaned_places.sort(key=lambda x: x["distance_km"])
    return cleaned_places[:limit]


# ===== Step 5: Select the best station based on weighted travel time (Distance Matrix) =====
def distance_matrix_times_minutes(origins, destinations, mode="transit", api_key=SERVER_KEY, transit_pref_api=None):
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
        "key": api_key,
        "language": "ko"
    }
    if mode == "transit" and transit_pref_api:
        params["transit_routing_preference"] = transit_pref_api
    
    try:
        r = requests.get(url, params=params, timeout=(5,15)).json()
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

def best_station_by_time(origins, stations, lam, mode="transit", api_key=SERVER_KEY, transit_pref_api=None):
    """
    Finds the best station from a list of candidates based on the weighted travel
    time from all origins to that station.
    """
    if not stations:
        return None

    try:
        times_mat = distance_matrix_times_minutes(origins, stations, mode=mode, api_key=api_key, transit_pref_api=transit_pref_api)
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

# ===== Main Pipeline Function =====
def pipeline(origins, lam=LAMBDA, grid_spacing_deg=GRID_SPACING_DEG, margin_deg=MARGIN_DEG,
             travel_mode="transit", api_key=SERVER_KEY, transit_pref_api=None):
    """
    Orchestrates the entire process of finding the optimal meeting point,
    with fallbacks to bus terminals or direct cafes if transit stations are not found.
    """
    result_type = None 
    found_main_place = None 
    cafes_list = []
    study_cafes_list = []

    # Step 1-3: Find best geographical point based on distance
    best_point = best_point_by_distance(origins,
                                        make_grid_candidates_fixed_count(origins, target_points=10000, margin_deg=margin_deg),
                                        lam=lam)
    if not best_point:
        print("Could not find a best geographical point.")
        return {"result_type": "error", "message": "유효한 출발 지점에서 중심 지점을 찾을 수 없습니다."}

    # --- Try 1: Find best Transit Station (Subway/Train) ---
    print("Trying to find best transit station...")
    stations, used_station_radius = get_transit_stations_within_radius_expand(
        best_point["lat"], best_point["lng"], api_key=api_key,
        start_km=1, max_km=10, min_count=5 # Max radius now 10km
    )
    best_station = best_station_by_time(origins, stations, lam=lam, mode=travel_mode, api_key=api_key)

    if best_station:
        result_type = "transit_station"
        found_main_place = best_station
        # Get cafes and study cafes separately by type and keyword
        cafes_list = get_nearby_places_sorted_by_distance(
            best_station["lat"], best_station["lng"], [{"type": "cafe"}], radius_m=1000, limit=5, api_key=api_key
        )
        # Search study cafes by keyword '스터디' in point_of_interest type
        study_cafes_list = get_nearby_places_sorted_by_distance(
            best_station["lat"], best_station["lng"], [{"type": "point_of_interest", "keyword": "스터디"}], radius_m=1000, limit=5, api_key=api_key
        )
        print(f"Found best transit station: {found_main_place['name']}")
    else:
        print("No best transit station found. Trying bus terminals...")
        # --- Try 2: Find best Bus Terminal if no Transit Station ---
        best_bus_terminal, used_bus_radius = get_best_bus_terminal(
            best_point["lat"], best_point["lng"], api_key=api_key, start_km=1, max_km=5
        )

        if best_bus_terminal:
            result_type = "bus_terminal"
            found_main_place = best_bus_terminal
            # Get cafes and study cafes separately for bus terminal
            cafes_list = get_nearby_places_sorted_by_distance(
                best_bus_terminal["lat"], best_bus_terminal["lng"], [{"type": "cafe"}], radius_m=1000, limit=5, api_key=api_key
            )
            # Search study cafes by keyword '스터디' in point_of_interest type
            study_cafes_list = get_nearby_places_sorted_by_distance(
                best_bus_terminal["lat"], best_bus_terminal["lng"], [{"type": "point_of_interest", "keyword": "스터디"}], radius_m=1000, limit=5, api_key=api_key
            )
            print(f"Found best bus terminal: {found_main_place['name']}")
        else:
            print("No bus terminal found. Falling back to direct cafes/study cafes around best geographical point.")
            # --- Try 3: Fallback to direct Cafes/Study Cafes around best_point ---
            result_type = "direct_places"
            found_main_place = best_point # In this case, best_point is the meeting point center
            # Get cafes and study cafes separately for direct places
            cafes_list = get_nearby_places_sorted_by_distance(
                best_point["lat"], best_point["lng"], [{"type": "cafe"}], radius_m=1000, limit=5, api_key=api_key
            )
            # Search study cafes by keyword '스터디' in point_of_interest type
            study_cafes_list = get_nearby_places_sorted_by_distance(
                best_point["lat"], best_point["lng"], [{"type": "point_of_interest", "keyword": "스터디"}], radius_m=1000, limit=5, api_key=api_key
            )
            if not cafes_list and not study_cafes_list:
                print("No cafes or study cafes found near the best geographical point.")
                result_type = "no_places_found" # Ultimate fallback if nothing is found

    final_result = {
        "result_type": result_type,
        "best_point": best_point, # This is the initial geographical best point
        "best_station": found_main_place if result_type == "transit_station" else None,
        "best_bus_terminal": found_main_place if result_type == "bus_terminal" else None,
        "direct_nearby_places_cafes": cafes_list if result_type == "direct_places" else [],
        "direct_nearby_places_study_cafes": study_cafes_list if result_type == "direct_places" else [],
        "cafes_list": cafes_list if result_type == "transit_station" or result_type == "bus_terminal" else [],
        "study_cafes_list": study_cafes_list if result_type == "transit_station" or result_type == "bus_terminal" else [],
        "message": "" 
    }
    
    return final_result

# ---------- [예약/상세 API] ----------

from flask import Response
from datetime import datetime
import pathlib

@app.route('/reserve')
def reserve_page():
    # templates/reservation.html 렌더 (쿼리스트링 place_id는 프런트 JS에서 읽음)
    return render_template('reservation.html')

@app.route('/api/place_details')
def api_place_details():
    """ Google Places Details → 가게 상세(가격대, 사진, 평점 등) 프록시 """
    place_id = request.args.get('place_id')
    if not place_id:
        return jsonify({"error": "place_id가 필요합니다."}), 400

    fields = ",".join([
        "name","formatted_address","international_phone_number",
        "website","rating","user_ratings_total","price_level",
        "opening_hours","photos","geometry","types"
    ])
    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&fields={fields}&language=ko&key={SERVER_KEY}"
    )
    r = requests.get(url, timeout=8)
    data = r.json()
    if data.get("status") != "OK":
        return jsonify({"error": data.get("status"), "detail": data}), 502

    result = data["result"]
    # 사진은 photo_reference만 전달하고, 이미지는 별도 /api/place_photo로
    photos = []
    for p in result.get("photos", [])[:8]:
        photos.append({
            "ref": p.get("photo_reference"),
            "width": p.get("width"),
            "height": p.get("height")
        })

    return jsonify({
        "name": result.get("name"),
        "address": result.get("formatted_address"),
        "phone": result.get("international_phone_number"),
        "website": result.get("website"),
        "rating": result.get("rating"),
        "ratings_total": result.get("user_ratings_total"),
        "price_level": result.get("price_level"),  # 0~4
        "opening_hours": result.get("opening_hours", {}),
        "location": result.get("geometry", {}).get("location"),
        "types": result.get("types", []),
        "photos": photos
    })

@app.route('/api/place_photo')
def api_place_photo():
    """ photo_reference → 이미지 바이너리 프록시 (브라우저에서 <img src>로 사용) """
    photoref = request.args.get("photoref")
    maxwidth = request.args.get("maxwidth", "800")
    if not photoref:
        return jsonify({"error": "photoref가 필요합니다."}), 400
    url = (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={maxwidth}&photo_reference={photoref}&key={SERVER_KEY}"
    )
    r = requests.get(url, stream=True, timeout=10)
    # Google이 302로 실제 이미지 CDN을 주는 경우가 있어 follow_redirects 역할
    if r.status_code in (301, 302) and 'location' in r.headers:
        r = requests.get(r.headers['location'], stream=True, timeout=10)
    content_type = r.headers.get('Content-Type', 'image/jpeg')
    return Response(r.content, content_type=content_type)

@app.route('/api/reservations', methods=['POST'])
def api_reservations():
    """ 예약 저장 (데모: 로컬 JSON에 append) """
    payload = request.get_json(force=True, silent=True) or {}
    required = ["place_id","place_name","date","time","party_size","name","phone"]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        return jsonify({"error": f"누락 필드: {', '.join(missing)}"}), 400

    # 간단한 ID 생성
    rid = f"rsv_{int(time.time())}"
    payload["reservation_id"] = rid
    payload["created_at"] = datetime.utcnow().isoformat() + "Z"

    data_dir = pathlib.Path("data"); data_dir.mkdir(exist_ok=True)
    db_file = data_dir / "reservations.json"
    if db_file.exists():
        try:
            db = json.loads(db_file.read_text(encoding="utf-8"))
        except Exception:
            db = []
    else:
        db = []
    db.append(payload)
    db_file.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "reservation_id": rid})








if __name__ == "__main__":
    print(" * Flask app starting on http://0.0.0.0:5000")
    print(" * Access it from your computer at http://127.0.0.1:5000 or http://localhost:5000")
    print(" * To access from other devices on your LOCAL NETWORK (e.g., phone on same Wi-Fi):")
    print("    1. Find your computer's local IP address (e.g., run 'ipconfig' on Windows or 'ifconfig' on macOS/Linux).")
    print("    2. On your device, open a browser and go to http://YOUR_COMPUTERS_IP:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
