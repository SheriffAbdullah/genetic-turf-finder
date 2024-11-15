import streamlit as st
import numpy as np
from genetic_algorithm import find_optimal_turf, haversine_distance

import folium
from streamlit_folium import st_folium

# Title and Introduction
st.title("Genetic Turf Finder")
st.markdown("""
This app uses a **genetic algorithm** to determine the best turf location.
""")

# Sidebar Legend
st.sidebar.markdown("""
### Map Legend:
- 🟦 **User Locations**
- 🟩 **Available Turfs**
- 🟪 **Best Turf**
- 🟧 **Outliers**
""")

# Default User and Turf Locations
user_locations = [
    (13.0300, 80.2700),  # Mylapore
    (13.0450, 80.2480),  # T Nagar
    (13.0125, 80.2500),  # Adyar
    (12.9800, 80.2200),  # Velachery
    (13.0850, 80.2100),  # Anna Nagar
    (13.2000, 80.1700),  # Outlier: Thiruvallur
]
turf_locations = [
    (13.0350, 80.2600),  # Turf 1
    (13.0100, 80.2400),  # Turf 2
    (13.0500, 80.2500),  # Turf 3
    (12.9900, 80.2300),  # Turf 4
    (13.0700, 80.2200),  # Turf 5
]

# Initialize session state
if "graph_ready" not in st.session_state:
    st.session_state["graph_ready"] = False

if "results" not in st.session_state:
    st.session_state["results"] = None

# User Input for Optimization Options
st.markdown("### Select Strategy")
option = st.radio(
    "Optimization Strategy",
    ("Minimize Overall Distance", "Prioritize a User"),
    help="Choose to minimize total distance or prioritize a specific user's proximity."
)

priority_user = None
if option == "Prioritize a User":
    priority_user = st.number_input(
        "Priority User Index",
        min_value=0,
        max_value=len(user_locations) - 1,
        value=0,
        step=1,
        help="Select the user to prioritize."
    )

# Outlier Discounting Options
enable_outlier_discounting = st.checkbox(
    "Enable Outlier Discounting",
    help="If enabled, users identified as outliers will be excluded in a second optimization step."
)

if enable_outlier_discounting:
    outlier_threshold = st.slider(
        "Outlier Threshold (multiples of standard deviation)",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Adjust the threshold for identifying outliers."
    )

# Run Genetic Algorithm and Display Results
if st.button("Find Optimal Turf"):
    try:
        # Run the genetic algorithm with optional outlier discounting
        results = find_optimal_turf(
            user_locations=user_locations,
            turfs=turf_locations,
            priority_user=priority_user if option == "Prioritize a User" else None,
            enable_outlier_discounting=enable_outlier_discounting
        )
        st.session_state["graph_ready"] = True
        st.session_state["results"] = results

        # Display initial and re-optimized results
        st.markdown("### Results Summary")
        st.markdown(f"**Initial Best Turf Location:** {results['initial_best_turf']}")
        if enable_outlier_discounting:
            st.markdown(f"**New Best Turf Location (Excluding Outliers):** {results['new_best_turf']}")
            st.markdown("#### Outliers:")
            for idx, outlier in enumerate(results["outliers"]):
                st.markdown(f"- Outlier {idx}: {outlier}")
        else:
            st.markdown("Outlier discounting is disabled.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display Map Visualization if Ready
if st.session_state["graph_ready"]:
    results = st.session_state["results"]

    # Calculate the average location to center the map
    all_locations = user_locations + turf_locations
    avg_lat = np.mean([loc[0] for loc in all_locations])
    avg_lon = np.mean([loc[1] for loc in all_locations])

    # Map Visualization
    st.markdown("### Map Visualization")
    map_ = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)  # Adjusted zoom level

    # Add user markers
    for idx, user in enumerate(user_locations):
        color = "orange" if user in results["outliers"] else "blue"
        popup = f"User {idx}" + (" (Outlier)" if user in results["outliers"] else "")
        folium.Marker(
            location=user,
            popup=popup,
            icon=folium.Icon(color=color)
        ).add_to(map_)

    # Add turf markers
    initial_best_turf = np.array(results["initial_best_turf"])
    new_best_turf = np.array(results.get("new_best_turf", []))

    for idx, turf in enumerate(turf_locations):
        color = "green" if (
            not np.array_equal(np.array(turf), initial_best_turf) and 
            not np.array_equal(np.array(turf), new_best_turf)
        ) else "purple"
        popup = f"Turf {idx}" + (
            " (Initial Best)" if np.array_equal(np.array(turf), initial_best_turf) else
            " (New Best)" if np.array_equal(np.array(turf), new_best_turf) else ""
        )
        folium.Marker(
            location=turf,
            popup=popup,
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(map_)

    st_folium(map_, width=700, height=500)

# Footer
st.markdown("""
---
Developed as a demonstration for using Genetic Algorithms in turf location optimization with outlier discounting.
""")
