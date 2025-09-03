import streamlit as st
import pandas as pd
import googlemaps
import plotly.express as px
import os
import time

st.subheader("Where is my cheese?")

# Replace with your actual API key
API_KEY = 'AIzaSyDBCqpgNPoLD6cYWTmNz9I86wHLiaUraG8'
gmaps = googlemaps.Client(key=API_KEY)

# Load addresses from Excel file
excel_path = r"C:\Users\thoma\OneDrive\Lebenslauf\_DB\_base_case\lat-long\convert to lat-lon.xlsx"
cache_path = r"C:\Users\thoma\OneDrive\Lebenslauf\_DB\_base_case\lat-long\geocode_cache.csv"

try:
    df_excel = pd.read_excel(excel_path, sheet_name="Sheet1")
    addresses = df_excel["Residenza"].dropna().unique().tolist()
except Exception as e:
    st.error(f"Error loading Excel file: {e}")
    addresses = []

# Load or create geocode cache
if os.path.exists(cache_path):
    cache_df = pd.read_csv(cache_path)
else:
    cache_df = pd.DataFrame(columns=["Address", "Latitude", "Longitude"])

# Geocode
results = []
cache_updated = False
cached_count = 0
api_count = 0

with st.spinner('Geocoding addresses...'):
    for i, address in enumerate(addresses):
        # Check if address is already in cache
        cached_result = cache_df[cache_df["Address"] == address]
        
        if not cached_result.empty:
            # Use cached coordinates
            row = cached_result.iloc[0]
            results.append((address, row["Latitude"], row["Longitude"]))
            cached_count += 1
        else:
            # Fetch from Google Maps API
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                loc = geocode_result[0]['geometry']['location']
                lat, lng = loc['lat'], loc['lng']
                results.append((address, lat, lng))
                
                # Add to cache
                new_row = pd.DataFrame({"Address": [address], "Latitude": [lat], "Longitude": [lng]})
                cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                cache_updated = True
                api_count += 1
            else:
                results.append((address, None, None))

# Display summary that disappears after 1 second
summary_placeholder = st.empty()
summary_placeholder.info(f"Geocoding complete! Used {cached_count} cached addresses and fetched {api_count} new addresses from Google Maps API.")
time.sleep(2)
summary_placeholder.empty()

# Save updated cache
if cache_updated:
    cache_df.to_csv(cache_path, index=False)
    st.success(f"Updated geocode cache with new addresses!")

# Merge on 'Residenza' (Excel) and 'Address' (cache)
merged_df = pd.merge(df_excel, cache_df, left_on="Residenza", right_on="Address", how="inner")

# Drop rows with missing coordinates
plot_df = merged_df.dropna(subset=['Latitude', 'Longitude']).copy()

if not plot_df.empty:
    # --- Plotly Map ---
    fig = px.scatter_mapbox(
        plot_df,
        lat="Latitude",
        lon="Longitude",
        # hover_name="Cliente",  # Main hover label
        # hover_data=[
        #     "NDG", "Nato a", "Nato il", "Cod Fisc", "Residenza", "Documento", "Telefono",
        #     "Appropriatezza", "Adeguatezza", "Profilazione", "Gestore", "Cod Gest", "Sportello",
        #     "Tel casa", "E-mail casa", "Tel lavoro", "E-mail lavoro", "Tel altro", "E-mail altro"
        # ],
        color_discrete_sequence=["blue"],
        zoom=5,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)

    # Display DataFrame in expander below the map
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    with st.expander("View Address Data"):
        st.dataframe(merged_df)

else:
    st.write("No coordinates to display on the map.")
    
    # Still show the dataframe even if no coordinates for mapping
    with st.expander("View Address Data"):
        st.dataframe(merged_df)
