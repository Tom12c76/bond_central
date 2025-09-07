import streamlit as st
st.set_page_config(page_title="Test", page_icon="🧪")


st.header("Tool Test Page")

import folium

# Coordinates from your JSON
lat, lon = 41.8955022, 12.4494833

# Create map centered on the location
m = folium.Map(location=[lat, lon], zoom_start=17)

# Add marker
folium.Marker(
    location=[lat, lon],
    popup="Via dell'Argilla, 4, 00165 Roma RM, Italy",
    tooltip="Click for address"
).add_to(m)

# Save map to HTML
m.save("rome_address_map.html")

print("Map saved as rome_address_map.html")