"""
Custom function tools.
"""

def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Simulate getting weather for a location."""
    # Mock weather data
    weather_data = {
        "New York": {"temp": 72, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Cloudy"},
        "Tokyo": {"temp": 25, "condition": "Rainy"},
        "Paris": {"temp": 18, "condition": "Partly Cloudy"},
    }
    data = weather_data.get(location, {"temp": 70, "condition": "Unknown"})
    return {
        "location": location,
        "temperature": data["temp"],
        "unit": unit,
        "condition": data["condition"]
    }

def get_time(location: str) -> dict:
    """Simulate getting current time for a location."""
    # Mock time data
    times = {
        "New York": "10:30 AM EST",
        "London": "3:30 PM GMT",
        "Tokyo": "11:30 PM JST",
        "Paris": "4:30 PM CET",
    }
    return {
        "location": location,
        "time": times.get(location, "12:00 PM UTC")
    }

def calculate_distance(from_location: str, to_location: str) -> dict:
    """Simulate calculating distance between two locations."""
    # Mock distance data (in miles)
    distances = {
        ("New York", "London"): 3459,
        ("New York", "Paris"): 3625,
        ("London", "Paris"): 214,
        ("Tokyo", "New York"): 6737,
    }
    key = (from_location, to_location)
    reverse_key = (to_location, from_location)
    distance = distances.get(key) or distances.get(reverse_key, 0)

    return {
        "from": from_location,
        "to": to_location,
        "distance_miles": distance,
        "distance_km": round(distance * 1.60934, 2)
    }
