import math
import mysql.connector
import folium
import webbrowser
import heapq
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Define global text box variables
text_box1 = None
text_box2 = None

# Define constants
SPEED = 26  # Speed in km/h
COST_PER_DAY = 1923.0  # Cost per day
EARTH_RADIUS_KM = 6371.0  # Radius of the Earth in kilometers

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
            landmark = None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        landmark = None
    finally:
        cursor.close()
        connection.close()
    return landmark

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two points on the Earth specified in decimal degrees."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
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

def a_star_search(grid, start, goal, checkpoints=None):
    """A* pathfinding algorithm to find a path avoiding land."""
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def neighbors(node):
        (x, y) = node
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
                yield new_x, new_y

    def reconstruct_path(came_from, start, goal):
        path = []
        current = goal
        if current in came_from:
            while current != start:
                path.append(current)
                current = came_from.get(current, start)
            path.append(start)
            path.reverse()
        return path

    # A* search algorithm with optional checkpoints
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    # Define checkpoint list
    checkpoint_list = [start] + (checkpoints or []) + [goal]

    full_path = []

    for i in range(len(checkpoint_list) - 1):
        current_start = checkpoint_list[i]
        current_goal = checkpoint_list[i + 1]

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == current_goal:
                segment_path = reconstruct_path(came_from, current_start, current_goal)
                full_path.extend(segment_path[1:])  # Avoid duplicating the checkpoint point
                break

            for next in neighbors(current):
                if grid[next] == 0:  # Ensure the cell is sea
                    new_cost = cost_so_far[current] + 1
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + heuristic(current_goal, next)
                        heapq.heappush(open_set, (priority, new_cost, next))
                        came_from[next] = current

        if not full_path:
            print(f"No path found from {current_start} to {current_goal}.")
            return []

        if i < len(checkpoint_list) - 2:
            # Reinitialize for the next segment
            open_set = []
            heapq.heappush(open_set, (0 + heuristic(full_path[-1], checkpoint_list[i + 2]), 0, full_path[-1]))
            came_from = {}
            cost_so_far = {}
            came_from[full_path[-1]] = None
            cost_so_far[full_path[-1]] = 0

    return full_path

def lat_lon_to_grid(lat: float, lon: float, lat_min: float, lon_min: float, lat_max: float, lon_max: float, grid_size: tuple) -> tuple:
    """Convert latitude and longitude to grid coordinates."""
    lat_ratio = (lat - lat_min) / (lat_max - lat_min)
    lon_ratio = (lon - lon_min) / (lon_max - lon_min)

    grid_x = int(lat_ratio * (grid_size[0] - 1))
    grid_y = int(lon_ratio * (grid_size[1] - 1))

    # Ensure grid_x and grid_y are within bounds
    grid_x = max(0, min(grid_x, grid_size[0] - 1))
    grid_y = max(0, min(grid_y, grid_size[1] - 1))

    return (grid_x, grid_y)

def grid_to_lat_lon(grid_x: int, grid_y: int, lat_min: float, lon_min: float, lat_max: float, lon_max: float, grid_size: tuple) -> tuple:
    """Convert grid coordinates to latitude and longitude."""
    lat = lat_min + (lat_max - lat_min) * grid_x / (grid_size[0] - 1)
    lon = lon_min + (lon_max - lon_min) * grid_y / (grid_size[1] - 1)
    return (lat, lon)

def create_map(lat1: float, lon1: float, lat2: float, lon2: float, name1: str, name2: str, path, grid, lat_min, lon_min, lat_max, lon_max):
    """Create a Folium map with markers and the sea route."""
    # Define the bounding box for the map
    bottom_left = [min(lat1, lat2), min(lon1, lon2)]
    top_right = [max(lat1, lat2), max(lon1, lon2)]

    # Create a base map centered around the midpoint
    map_center = [(bottom_left[0] + top_right[0]) / 2, (bottom_left[1] + top_right[1]) / 2]
    m = folium.Map(location=map_center, zoom_start=12)

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
    
    # Convert grid path to latitude/longitude
    if path:
        path_locations = []
        for p in path:
            lat_lon = grid_to_lat_lon(p[0], p[1], lat_min, lon_min, lat_max, lon_max, grid.shape)
            path_locations.append(lat_lon)
        
        folium.PolyLine(locations=path_locations, color='green').add_to(m)

    # Save the map to an HTML file
    map_filename = 'route_map.html'
    m.save(map_filename)
    webbrowser.open(map_filename)

def show_results(distance, travel_time, cost):
    """Show a new window with the distance, travel time, and cost."""
    results_window = tk.Toplevel(root)
    results_window.geometry('1500x1000')

    # Load and set the background image for the new window
    bg_image = Image.open(r"C:\Users\thaku\Documents\Fsh\jj.png")
    bg_image = bg_image.resize((1500, 1000), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Keep a reference to the image
    results_window.bg_image = bg_photo

    # Create and place a canvas on the new window
    canvas = tk.Canvas(results_window, width=1500, height=1000)
    canvas.pack(fill="both", expand=True)

    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Display the results
    results_text = (
        f"Distance: {distance:.2f} km\n"
        f"Travel Time: {travel_time:.2f} hours\n"
        f"Cost: ${cost:.2f}"
    )
    
    results_label = tk.Label(results_window, text=results_text, font=("Arial", 24, "bold"), bg="#ffffff", fg="#000000")
    canvas.create_window(750, 500, window=results_label, anchor="n")

def on_submit():
    # Retrieve the values from the textboxes
    name1 = text_box1.get()
    name2 = text_box2.get()
    
    # Fetch landmarks from MySQL database
    landmark1 = fetch_landmark_by_name(name1)
    landmark2 = fetch_landmark_by_name(name2)

    if not landmark1 or not landmark2:
        print("One or both landmarks not found.")
        return

    # Extract latitude and longitude
    lat1, lon1 = float(landmark1['x']), float(landmark1['y'])
    lat2, lon2 = float(landmark2['x']), float(landmark2['y'])

    # Define grid size and bounds
    grid_size = (100, 100)  # Define the size of the grid
    grid = np.zeros(grid_size)  # Initialize grid as sea
    grid[30:40, 20:30] = 1  # Example land area (you should use real land data)
    lat_min, lon_min = 48.0, 2.0  # These should match your grid's geographic bounds
    lat_max, lon_max = 49.0, 3.0  # These should match your grid's geographic bounds

    # Convert latitude and longitude to grid coordinates
    start_grid = lat_lon_to_grid(lat1, lon1, lat_min, lon_min, lat_max, lon_max, grid_size)
    goal_grid = lat_lon_to_grid(lat2, lon2, lat_min, lon_min, lat_max, lon_max, grid_size)

    # Perform pathfinding
    path = a_star_search(grid, start_grid, goal_grid)

    # Calculate distance, travel time, and cost
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    travel_time = time_between_ports(distance)
    cost = cost_of_travel(travel_time)

    # Create and save the map with the sea route and markers
    create_map(lat1, lon1, lat2, lon2, name1, name2, path, grid, lat_min, lon_min, lat_max, lon_max)

    # Show the results in a new window
    show_results(distance, travel_time, cost)

def on_button_click():
    # Create the new window
    new_window = tk.Toplevel(root)
    new_window.geometry('1500x1000')

    # Load and set the background image for the new window
    bg_image = Image.open(r"C:\Users\thaku\Documents\Fsh\jj.png")
    bg_image = bg_image.resize((1500, 1000), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Keep a reference to the image
    new_window.bg_image = bg_photo

    # Create and place a canvas on the new window
    canvas = tk.Canvas(new_window, width=1500, height=1000)
    canvas.pack(fill="both", expand=True)

    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Add titles above the textboxes
    title1 = tk.Label(new_window, text="Enter the Origin port", font=("Arial", 22, "bold"), bg="#ffffff", fg="#000000")
    canvas.create_window(220, 200, window=title1, anchor="n")  # Adjust position as needed

    title2 = tk.Label(new_window, text="Enter the End port", font=("Arial", 22, "bold"), bg="#ffffff", fg="#000000")
    canvas.create_window(220, 400, window=title2, anchor="n")  # Adjust position as needed

    # Add two textboxes (entries) to the new window
    global text_box1
    text_box1 = tk.Entry(new_window, font=("Arial", 18), width=30, borderwidth=2, relief="solid")
    canvas.create_window(220, 250, window=text_box1, anchor="n")  # Adjust position as needed

    global text_box2
    text_box2 = tk.Entry(new_window, font=("Arial", 18), width=30, borderwidth=2, relief="solid")
    canvas.create_window(220, 450, window=text_box2, anchor="n")  # Adjust position as needed

    # Add a button that will call the on_submit function
    submit_button = tk.Button(new_window, text="Find!!", command=on_submit, font=("Arial", 18), width=20, height=2)
    canvas.create_window(750, 600, window=submit_button, anchor="n")  # Adjust position as needed

# Your main code for the root window setup
root = tk.Tk()
root.title('Fsh')
root.geometry('1500x1000')

# Background 1st slide
bg_image = Image.open(r"C:\Users\thaku\Documents\Fsh\Start.png")
bg_image = bg_image.resize((1500, 1000), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Keep a reference to the image
root.bg_image = bg_photo

canvas1 = tk.Canvas(root, width=1500, height=1000)
canvas1.pack(fill="both", expand=True)

canvas1.create_image(0, 0, image=bg_photo, anchor="nw")

# Slide-1 button
style = ttk.Style()
style.configure("Modern.TButton",
                padding=30,
                relief="groove",
                background="#2196f3",
                foreground="blue",
                activebackground="#65e7ff",
                activeforeground="#65e7ff",
                font=("Arial", 17, "bold"))

modern_button = ttk.Button(root, text="Start Here", style="Modern.TButton", command=on_button_click)
canvas1.create_window(750, 550, window=modern_button)

# Text 1st slide
text_label = tk.Label(root, text="FSH-Optimal Ship Routes", bg="#0A2472", fg="#ffffff", font=("Arial", 55, "bold"))
canvas1.create_window(750, 250, window=text_label)

root.mainloop()

