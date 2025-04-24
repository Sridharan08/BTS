from flask import Flask, jsonify, request
from ultralytics import YOLO
from flask_cors import CORS
from twilio.rest import Client
import os
import googlemaps
from dotenv import load_dotenv
import time
from flask_socketio import SocketIO, emit
import requests
from pymongo import MongoClient
from bson.json_util import dumps


app = Flask(__name__)
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app)

# Load environment variables
load_dotenv()

# Constants
TOTAL_SEATS = 48
IMAGE_DIR = "images"
OUTPUT_DIR = "static"

# Load YOLO model
MODEL_PATH = "yolov5/yolov5lu.pt"
model = YOLO(MODEL_PATH)

# Twilio Setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
TO_NUMBER = os.getenv("TO_NUMBER")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Google Maps setup
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

# MongoDB Setup
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client['bus_tracking']
bus_collection = db['buses']
search_history_collection = db['search_history']

# Storage for location and dashboard
bus_data = {
    "current_location": {"latitude": None, "longitude": None},
    "route": [],
    "last_updated": None
}

search_history = []

# Define all bus metadata
buses = {
    "86B": {
        "from": "Gandhipuram",
        "to": "Ukkadam",
        "from_coords": {"lat": 11.0168, "lng": 76.9558},
        "to_coords": {"lat": 10.9925, "lng": 76.9614},
        "stops": [
            {"name": "Town Hall", "lat": 11.0123, "lng": 76.9567},
            {"name": "Railway Station", "lat": 11.0056, "lng": 76.9589}
        ],
        "image_path": "images/86B.jpg",
        "detected_image": "static/86B_detected.jpg",
        "schedule": ["07:00", "09:00", "12:00", "15:00", "18:00"]
    },
    "20C": {
        "from": "Ukkadam",
        "to": "Kuniyamuthur",
        "from_coords": {"lat": 10.9925, "lng": 76.9614},
        "to_coords": {"lat": 10.9786, "lng": 76.9483},
        "stops": [
            {"name": "Peelamedu", "lat": 10.9854, "lng": 76.9552},
            {"name": "PSG Tech", "lat": 10.9817, "lng": 76.9521}
        ],
        "image_path": "images/20C.jpg",
        "detected_image": "static/20C_detected.jpg",
        "schedule": ["06:30", "08:30", "11:30", "14:30", "17:30"]
    },
    "1A": {
        "from": "Gandhipuram",
        "to": "Peelamedu",
        "from_coords": {"lat": 11.0168, "lng": 76.9558},
        "to_coords": {"lat": 10.9854, "lng": 76.9552},
        "stops": [
            {"name": "RS Puram", "lat": 11.0089, "lng": 76.9567},
            {"name": "Race Course", "lat": 10.9967, "lng": 76.9578}
        ],
        "image_path": "images/bus1.jpg",
        "detected_image": "static/1A_detected.jpg",
        "schedule": ["07:15", "09:15", "12:15", "15:15", "18:15"]
    },
    "S9": {
        "from": "Ukkadam",
        "to": "Singanallur",
        "from_coords": {"lat": 10.9925, "lng": 76.9614},
        "to_coords": {"lat": 11.0004, "lng": 77.0122},
        "stops": [
            {"name": "Podanur", "lat": 10.9824, "lng": 76.9742},
            {"name": "Ramanathapuram", "lat": 10.9963, "lng": 76.9914}
        ],
        "image_path": "images/bus2.webp",
        "detected_image": "static/S9_detected.jpg",
        "schedule": ["06:45", "09:00", "11:15", "13:45", "17:15"]
    },
    "3E": {
        "from": "Gandhipuram",
        "to": "Saravanampatti",
        "from_coords": {"lat": 11.0168, "lng": 76.9558},
        "to_coords": {"lat": 11.0812, "lng": 76.9949},
        "stops": [
            {"name": "Peelamedu", "lat": 10.9854, "lng": 76.9552},
            {"name": "Hope College", "lat": 11.0225, "lng": 76.9967}
        ],
        "image_path": "images/bus3.webp",
        "detected_image": "static/3E_detected.jpg",
        "schedule": ["06:00", "08:15", "10:30", "13:00", "16:30"]
    }
}

# YOLO-based seat detection
def detect_empty_seats_for_image(image_path, detected_image_path):
    try:
        if not os.path.exists(image_path):
            return {"error": "Image not found", "empty_seats": 0, "status": "Error", "image_url": None}

        results = model(image_path)
        person_count = sum(
            1 for result in results for box in result.boxes if int(box.cls[0]) == 0
        )
        empty_seats = max(TOTAL_SEATS - person_count, 0)

        for result in results:
            result.save(detected_image_path)

        status = "Bus is almost full" if empty_seats < 10 else "Bus has plenty of seats" if empty_seats > 30 else "Normal occupancy"
        
        return {
            "error": None,
            "empty_seats": empty_seats,
            "status": status,
            "image_url": f"/{detected_image_path}"
        }
    except Exception as e:
        return {"error": str(e), "empty_seats": 0, "status": "Error", "image_url": None}

# GET buses from -> to with seat info
@app.route('/api/bus', methods=['GET'])
def get_bus_details():
    from_location = request.args.get('from')
    to_location = request.args.get('to')

    if not from_location or not to_location:
        return jsonify({"error": "Missing 'from' or 'to' location parameters"}), 400

    matched_buses = []

    for bus_number, bus in buses.items():
        if bus['from'].lower() == from_location.lower() and bus['to'].lower() == to_location.lower():
            seat_data = detect_empty_seats_for_image(bus["image_path"], bus["detected_image"])
            if seat_data["error"]:
                continue

            matched_buses.append({
                "busNumber": bus_number,
                "from": bus["from"],
                "to": bus["to"],
                "from_coords": bus["from_coords"],
                "to_coords": bus["to_coords"],
                "stops": bus.get("stops", []),
                "emptySeats": seat_data["empty_seats"],
                "status": seat_data["status"],
                "image_url": seat_data["image_url"],
                "schedule": bus.get("schedule", [])
            })

    search_history.append({
        "from": from_location,
        "to": to_location,
        "buses": [bus["busNumber"] for bus in matched_buses],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    if matched_buses:
        sms_text = f"🚌 Bus search from '{from_location}' to '{to_location}' found {len(matched_buses)} bus(es):\n"
        for bus in matched_buses:
            sms_text += f"➡ {bus['busNumber']} ({bus['status']}, {bus['emptySeats']} empty seats)\n"
        send_sms(sms_text)
    else:
        send_sms(f"🔍 No buses found from '{from_location}' to '{to_location}'.")

    return jsonify(matched_buses)

# POST location updates
@app.route('/api/location', methods=['POST'])
def update_location():
    data = request.get_json()
    try:
        lat = float(data.get("latitude"))
        lng = float(data.get("longitude"))
        bus_number = data.get("bus_number")

        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError("Invalid coordinates")

        bus_data["current_location"] = {"latitude": lat, "longitude": lng, "timestamp": time.time()}
        if len(bus_data["route"]) >= 10:
            bus_data["route"].pop(0)
        bus_data["route"].append({"lat": lat, "lng": lng})

        bus_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        send_sms(f"🚌 Bus {bus_number} location updated: 📍 {lat}, {lng} at {bus_data['last_updated']}")

        return jsonify({
            "message": "Location updated",
            "bus_number": bus_number,
            "timestamp": bus_data["last_updated"]
        }), 200
    except Exception as e:
        return jsonify({"error": "Invalid location data"}), 400

# GET current location
@app.route('/api/location', methods=['GET'])
def get_location():
    return jsonify({
        "current_location": bus_data["current_location"],
        "last_updated": bus_data["last_updated"],
        "route_history": bus_data["route"]
    }), 200

# GET single bus details
@app.route('/api/bus/<bus_number>', methods=['GET'])
def get_bus_info(bus_number):
    bus = buses.get(bus_number.upper())
    if not bus:
        return jsonify({"error": "Bus not found"}), 404

    return jsonify({
        "bus_number": bus_number,
        "from": bus["from"],
        "to": bus["to"],
        "from_coords": bus["from_coords"],
        "to_coords": bus["to_coords"],
        "stops": bus.get("stops", []),
        "schedule": bus.get("schedule", []),
        "current_location": bus_data["current_location"],
        "last_updated": bus_data["last_updated"]
    })

# Admin Dashboard API
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    total_buses = len(buses)

    # Route analysis
    route_counts = {}
    for entry in search_history:
        route_key = f"{entry['from']} - {entry['to']}"
        route_counts[route_key] = route_counts.get(route_key, 0) + 1

    busiest_routes = [{"route": route, "count": count} for route, count in route_counts.items()]
    busiest_routes.sort(key=lambda x: x["count"], reverse=True)

    # Hourly activity analysis
    hour_activity = {}
    for entry in search_history:
        hour = entry["timestamp"].split(" ")[1].split(":")[0]
        hour_activity[hour] = hour_activity.get(hour, 0) + 1

    peak_hours = [{"hour": f"{hour}:00", "activity": activity} for hour, activity in sorted(hour_activity.items())]

    # Speed and Delay Analysis
    speeds = []
    delays = []

    current_timestamp = time.time()
    for bus_number, bus in buses.items():
        route = bus_data["route"]
        if len(route) >= 2:
            start_coords = route[0]
            end_coords = route[-1]

            start_time_struct = time.strptime(bus_data["last_updated"], "%Y-%m-%d %H:%M:%S")
            start_time = time.mktime(start_time_struct)

            time_taken_hr = (current_timestamp - start_time) / 3600.0
            distance_km = calculate_distance(start_coords, end_coords)
            speed_kmph = distance_km / time_taken_hr if time_taken_hr else 0
            speeds.append(speed_kmph)

            # Estimate delay (assumes first scheduled time for simplicity)
            if bus.get("schedule"):
                scheduled_time_str = bus["schedule"][0]
                scheduled_struct = time.strptime(f"{time.strftime('%Y-%m-%d')} {scheduled_time_str}", "%Y-%m-%d %H:%M")
                scheduled_epoch = time.mktime(scheduled_struct)
                delay_minutes = max(0, (current_timestamp - scheduled_epoch) / 60)
                delays.append(delay_minutes)

    average_speed = sum(speeds) / len(speeds) if speeds else 0
    average_delay = sum(delays) / len(delays) if delays else 0

    traffic_status = (
        "Heavy" if average_delay > 20 else
        "Moderate" if average_delay > 10 else
        "Smooth"
    )

    dashboard_data = {
        "total_buses": total_buses,
        "busiest_routes": busiest_routes,
        "peak_hours": peak_hours,
        "average_speed_kmph": round(average_speed, 2),
        "average_delay_minutes": round(average_delay, 2),
        "traffic_status": traffic_status
    }

    return jsonify(dashboard_data)



def calculate_distance(start_coords, end_coords):
    """Calculate the distance between two coordinates (in kilometers)"""
    from geopy.distance import geodesic
    start = (start_coords['lat'], start_coords['lng'])
    end = (end_coords['lat'], end_coords['lng'])
    return geodesic(start, end).km


def calculate_delay(scheduled_time, current_time):
    from datetime import datetime
    scheduled = datetime.strptime(scheduled_time, "%H:%M")
    current = datetime.strptime(current_time, "%H:%M")
    delta = current - scheduled
    return max(0, delta.total_seconds() / 60)

# --- Geolocation Functions ---
def get_ip_geolocation():
    """Fallback to IP-based geolocation if GPS is unavailable."""
    try:
        response = requests.get("http://ip-api.com/json")
        data = response.json()
        if data["status"] == "success":
            return {
                "latitude": data["lat"],
                "longitude": data["lon"],
                "accuracy": 5000  # IP accuracy is low (~5km)
            }
    except Exception as e:
        print("IP geolocation failed:", e)
    return None

def get_google_geolocation():
    """High-accuracy location using Google Maps Geolocation API."""
    try:
        result = gmaps.geolocate()
        return {
            "latitude": result["location"]["lat"],
            "longitude": result["location"]["lng"],
            "accuracy": result["accuracy"]
        }
    except Exception as e:
        print("Google Geolocation error:", e)
        return None

# POST location updates
@app.route('/api/buslocation', methods=['POST'])
def post_bus_location():  # 🔄 renamed from update_location
    data = request.get_json()
    try:
        lat = float(data.get("latitude"))
        lng = float(data.get("longitude"))
        bus_number = data.get("bus_number")

        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError("Invalid coordinates")

        bus_data["current_location"] = {"latitude": lat, "longitude": lng, "timestamp": time.time()}
        if len(bus_data["route"]) >= 10:
            bus_data["route"].pop(0)
        bus_data["route"].append({"lat": lat, "lng": lng})

        bus_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        send_sms(f"🚌 Bus {bus_number} location updated: 📍 {lat}, {lng} at {bus_data['last_updated']}")

        return jsonify({
            "message": "Location updated",
            "bus_number": bus_number,
            "timestamp": bus_data["last_updated"]
        }), 200
    except Exception as e:
        return jsonify({"error": "Invalid location data"}), 400


# GET current location
@app.route('/api/location', methods=['GET'])
def get_location_data():
    return jsonify({
        "current_location": bus_data["current_location"],
        "last_updated": bus_data["last_updated"],
        "route_history": bus_data["route"]
    }), 200


# --- WebSocket Handlers ---
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('location_update', {
        "bus_number": "ALL",
        "location": bus_data["current_location"],
        "timestamp": bus_data["last_updated"]
    })

# SMS sender
def send_sms(message):
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=TO_NUMBER
        )
        print("✅ SMS sent.")
    except Exception as e:
        print("❌ SMS error:", e)

if __name__ == '_main_':
    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
    # app.run(debug=True, host='0.0.0.0', port=5000)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

