import streamlit as st
import pandas as pd
import googlemaps
import plotly.express as px
import os
import time
import json

st.set_page_config(page_title="Lat Long", page_icon="🌍", layout="wide")

st.subheader("Where is my cheese?")

# Replace with your actual API key
API_KEY = 'AIzaSyDBCqpgNPoLD6cYWTmNz9I86wHLiaUraG8'
gmaps = googlemaps.Client(key=API_KEY)

# Define a local cache path
cache_path = "temp_data/geocode_cache.csv"

# File uploader
uploaded_file = st.file_uploader("Upload your file (XLSX or CSV)", type=['xlsx', 'csv'])

def get_nested_value(d, keys, default=None):
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return default
    return d if d is not None else default

df_excel = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df_excel = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df_excel = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")

if df_excel is not None:
    st.success(f"File '{uploaded_file.name}' loaded successfully.")
    
    address_column = st.selectbox("Select the column containing addresses:", df_excel.columns, index=None)

    if address_column:
        addresses = df_excel[address_column].dropna().unique().tolist()

        # Define all possible columns for the cache
        address_component_types = [
            'street_number', 'route', 'locality', 'administrative_area_level_3',
            'administrative_area_level_2', 'administrative_area_level_1',
            'country', 'postal_code'
        ]
        address_component_columns = [f"{t}_long" for t in address_component_types] + \
                                    [f"{t}_short" for t in address_component_types]

        geometry_columns = [
            'bounds_ne_lat', 'bounds_ne_lng', 'bounds_sw_lat', 'bounds_sw_lng',
            'viewport_ne_lat', 'viewport_ne_lng', 'viewport_sw_lat', 'viewport_sw_lng'
        ]
        
        base_cache_columns = [
            "Address", "Latitude", "Longitude", "formatted_address",
            "place_id", "location_type", "types"
        ]
        cache_columns = base_cache_columns + geometry_columns + address_component_columns

        # Load or create geocode cache
        if os.path.exists(cache_path):
            cache_df = pd.read_csv(cache_path)
            # Add any missing columns to older cache files
            for col in cache_columns:
                if col not in cache_df.columns:
                    cache_df[col] = pd.NA
        else:
            cache_df = pd.DataFrame(columns=cache_columns)

        # Geocode
        cache_updated = False
        api_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner('Checking cache and geocoding addresses...'):
            for i, address in enumerate(addresses):
                progress = (i + 1) / len(addresses)
                progress_bar.progress(progress)
                status_text.text(f"Processing address {i+1}/{len(addresses)}: {address}")

                cached_result = cache_df[cache_df["Address"] == address]
                
                if cached_result.empty or pd.isna(cached_result.iloc[0].get("formatted_address")):
                    geocode_result = gmaps.geocode(address)
                    if geocode_result:
                        api_count += 1
                        res = geocode_result[0]
                        
                        new_data = {
                            "Address": address,
                            "Latitude": get_nested_value(res, ['geometry', 'location', 'lat']),
                            "Longitude": get_nested_value(res, ['geometry', 'location', 'lng']),
                            "formatted_address": res.get('formatted_address'),
                            "place_id": res.get('place_id'),
                            "location_type": get_nested_value(res, ['geometry', 'location_type']),
                            "types": ", ".join(res.get('types', [])),
                            
                            'bounds_ne_lat': get_nested_value(res, ['geometry', 'bounds', 'northeast', 'lat']),
                            'bounds_ne_lng': get_nested_value(res, ['geometry', 'bounds', 'northeast', 'lng']),
                            'bounds_sw_lat': get_nested_value(res, ['geometry', 'bounds', 'southwest', 'lat']),
                            'bounds_sw_lng': get_nested_value(res, ['geometry', 'bounds', 'southwest', 'lng']),
                            'viewport_ne_lat': get_nested_value(res, ['geometry', 'viewport', 'northeast', 'lat']),
                            'viewport_ne_lng': get_nested_value(res, ['geometry', 'viewport', 'northeast', 'lng']),
                            'viewport_sw_lat': get_nested_value(res, ['geometry', 'viewport', 'southwest', 'lat']),
                            'viewport_sw_lng': get_nested_value(res, ['geometry', 'viewport', 'southwest', 'lng']),
                        }

                        # Extract address components
                        for comp_type in address_component_types:
                            new_data[f"{comp_type}_long"] = None
                            new_data[f"{comp_type}_short"] = None

                        for component in res.get('address_components', []):
                            for comp_type in component.get('types', []):
                                if comp_type in address_component_types:
                                    new_data[f"{comp_type}_long"] = component.get('long_name')
                                    new_data[f"{comp_type}_short"] = component.get('short_name')
                        
                        if not cached_result.empty:
                            for key, value in new_data.items():
                                cache_df.loc[cached_result.index, key] = value
                        else:
                            new_row = pd.DataFrame([new_data])
                            cache_df = pd.concat([cache_df, new_row], ignore_index=True)
                        
                        cache_updated = True

            status_text.text("Geocoding complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

        cached_count = len(addresses) - api_count
        st.info(f"Geocoding complete! Used {cached_count} cached addresses and fetched {api_count} new addresses from Google Maps API.")

        if cache_updated:
            cache_df.to_csv(cache_path, index=False)
            st.success(f"Updated geocode cache with {api_count} new entries!")

        merged_df = pd.merge(df_excel, cache_df, left_on=address_column, right_on="Address", how="inner")
        plot_df = merged_df.dropna(subset=['Latitude', 'Longitude']).copy()

        if not plot_df.empty:
            st.subheader("Address Map")
            
            all_columns = merged_df.columns.tolist()
            default_hover = [col for col in [address_column, "country_long", "formatted_address"] if col in all_columns]
            hover_data = st.multiselect("Select data to show on hover:", all_columns, default=default_hover)

            fig = px.scatter_mapbox(
                plot_df,
                lat="Latitude",
                lon="Longitude",
                color="country_long",
                hover_name="formatted_address",
                hover_data=hover_data,
                zoom=5,
                height=600
            )

            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig)

            with st.expander("View Full Geocoded Data"):
                st.dataframe(merged_df)
        else:
            st.write("No coordinates to display on the map.")
            with st.expander("View Full Geocoded Data"):
                st.dataframe(merged_df)
else:
    st.info("Please upload a file to begin.")
