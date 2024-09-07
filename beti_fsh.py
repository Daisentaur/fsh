import math
import mysql.connector
import folium

# Define constants
SPEED = 26  # Speed in km/h
COST_PER_DAY = 1923.0  # Cost per day
EARTH_RADIUS_KM = 6371.0  # Radius of the Earth in kilometers

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two points on the Earth specified in decimal degrees."""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Compute differences
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_RADIUS_KM * c
    
    return distance

def time_between_ports(distance: float) -> float:
    """Calculate travel time between ports given the distance and speed."""
    if SPEED == 0:  # Avoid division by zero
        print("Speed cannot be zero.")
        return 0.0
    return distance / SPEED

def cost_of_travel(travel_time: float) -> float:
    """Calculate travel cost given the travel time in hours and cost per day."""
    days = travel_time / 24  # Convert hours to days
    return days * COST_PER_DAY

def fetch_landmark_by_name(name: str):
    """Fetch a landmark's coordinates from MySQL database by name."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='tiger123',
            database='FSH'
        )
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM landmarks WHERE port_name = %s"
        cursor.execute(query, (name,))
        landmark = cursor.fetchone()
        if not landmark:
            print(f"Landmark '{name}' not found.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        landmark = None
    finally:
        cursor.close()
        connection.close()
    return landmark

def create_map(lat1: float, lon1: float, lat2: float, lon2: float, name1: str, name2: str):
    """Create a Folium map with markers for the two landmarks."""
    # Create a base map
    map_center = [(lat1 + lat2) / 2, (lon1 + lon2) / 2]
    m = folium.Map(location=map_center, zoom_start=10)

    # Add markers for the landmarks
    folium.Marker(
        location=[lat1, lon1],
        popup=name1,
        icon=folium.Icon(color='blue')
    ).add_to(m)
    
    folium.Marker(
        location=[lat2, lon2],
        popup=name2,
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    # Optionally, you can add a line between the two landmarks
    folium.PolyLine(locations=[[lat1, lon1], [lat2, lon2]], color='green').add_to(m)
    
    # Save the map to an HTML file
    m.save('landmarks_map.html')
    
def main():
    # Take two names as input
    name1 = input("Enter the name of the Origin Port: ")
    name2 = input("Enter the name of the Destination Port: ")

    # Fetch landmarks from MySQL database
    landmark1 = fetch_landmark_by_name(name1)
    landmark2 = fetch_landmark_by_name(name2)

    if not landmark1 or not landmark2:
        print("One or both landmarks not found.")
        return

    # Extract latitude and longitude
    lat1, lon1 = float(landmark1['x']), float(landmark1['y'])
    lat2, lon2 = float(landmark2['x']), float(landmark2['y'])

    # Calculate distance between the two landmarks using Haversine formula
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    travel_time = time_between_ports(dist)
    travel_cost = cost_of_travel(travel_time)

    # Output results
    print(f'Distance between {name1} and {name2}: {dist:.2f} km')
    print(f'Travel Time: {travel_time:.2f} hours')
    print(f'Travel Cost: ${travel_cost:.2f}')

    create_map(lat1, lon1, lat2, lon2, name1, name2)

if __name__ == "__main__":
    main()
