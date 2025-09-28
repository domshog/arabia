import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from datetime import datetime
import re
import os
import math

# Set page configuration
st.set_page_config(
    page_title="Arabian Peninsula Renewable Energy Assessment",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Arabian Peninsula cities with coordinates
ARABIAN_CITIES = {
    "Saudi Arabia": {
        "Riyadh": {"lat": 24.539, "lon": 46.663},
        "Mecca": {"lat": 21.485, "lon": 39.763},
        "Jazan": {"lat": 17.247, "lon": 42.839}
    },
    "UAE": {
        "Dubai": {"lat": 24.981, "lon": 55.317}
    },
    "Kuwait": {
        "Kuwait": {"lat": 29.253, "lon": 47.731}
    },
    "Qatar": {
        "Doha": {"lat": 25.242, "lon": 51.437}
    },
    "Bahrain": {
        "Manama": {"lat": 26.079, "lon": 50.583}
    },
    "Oman": {
        "Muscat": {"lat": 23.615, "lon": 58.555},
        "Salalah": {"lat": 17.076, "lon": 54.093}
    },
    "Yemen": {
        "Sanaa": {"lat": 15.323, "lon": 44.191},
        "Aden": {"lat": 12.848, "lon": 45.032},
        "Al_Hudaydah": {"lat": 14.777, "lon": 42.977},
        "Sayun": {"lat": 15.967, "lon": 48.797}
        #"Al_Hudaydah": {"lat": 14.777, "lon": 42.977}
    },
    "Palastine": {
        "Gaza": {"lat": 31.506, "lon": 34.461},
        "AlQuds": {"lat": 31.788, "lon": 35.206}
    },
    "Iraq": {
        "Baghdad": {"lat": 33.305, "lon": 44.394},
    },
    "Syria": {
        "Damascus": {"lat": 33.537, "lon": 36.339},
        "Aleppo": {"lat": 36.197, "lon": 37.176}
    },
    
    "Jordan": {
        "Amman": {"lat": 31.954, "lon": 35.973},
        "Ma'an": {"lat": 30.554, "lon": 36.764}
    }
}

def fix_and_load_tmy_file(filepath):
    """
    Reads the malformed TMY file, inserts line breaks before each timestamp,
    and returns a properly formatted pandas DataFrame.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to split the content at each timestamp (YYYYMMDD:HHMM)
        pattern = r'(\d{8}:\d{4})'
        parts = re.split(pattern, content)
        
        lines = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                line = parts[i] + parts[i + 1]
                lines.append(line.strip())
            else:
                lines.append(parts[i].strip())
        
        data_rows = []
        for line in lines:
            row = [item.strip() for item in line.split(',') if item.strip()]
            if len(row) >= 10:
                data_rows.append(row[:10])
        
        column_names = [
            'time(UTC)', 'T2m', 'RH', 'G(h)', 'Gb(n)', 'Gd(h)', 'IR(h)', 'WS10m', 'WD10m', 'SP'
        ]
        
        df = pd.DataFrame(data_rows, columns=column_names)
        
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse datetime and create a display-friendly format (MMDD:HHMM)
        df['datetime'] = pd.to_datetime(df['time(UTC)'], format='%Y%m%d:%H%M')
        df['display_time'] = df['datetime'].dt.strftime('%m%d:%H%M')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['time(UTC)'], inplace=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file {filepath}: {e}")
        return None

def get_local_tmy_data(city_name, lat, lon, country_code=""):
    """Load and process local TMY data from a CSV file for the specified city."""
    filename_variants = [
        f"tmy_{lat}_{lon}_2005_2023_{city_name.lower()}.csv",
        f"{city_name.lower()}_{lat}_{lon}_2005_2023_{country_code.lower()}.csv",
        f"{city_name.lower()}.csv"
    ]
    
    df = None
    used_filename = None
    
    for filename in filename_variants:
        if os.path.exists(filename):
            try:
                df = fix_and_load_tmy_file(filename)
                if df is not None:
                    used_filename = filename
                    break
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
                continue
    
    if df is None:
        st.warning(f"No TMY data file found or could not be processed for {city_name}.")
        return None
    
    first_row = df.iloc[0] if len(df) > 0 else {}
    
    weather_dict = {
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {
            "temp": first_row.get('T2m', 35.0),
            "humidity": first_row.get('RH', 45.0),
            "pressure": first_row.get('SP', 101300) / 100.0
        },
        "wind": {
            "speed": first_row.get('WS10m', 4.5),
            "deg": first_row.get('WD10m', 180)
        },
        "clouds": {"all": 10},
        "sys": {
            "sunrise": 1640844000,
            "sunset": 1640883600
        },
        "tmy_dataframe": df
    }
    
    return weather_dict

def get_weather_data(lat, lon, city_name="Unknown", country_code=""):
    """Fetch weather data with TMY data priority."""
    tmy_data = get_local_tmy_data(city_name, lat, lon, country_code)
    if tmy_data is not None:
        return tmy_data

    # Enhanced sample data for Arabian Peninsula
    sample_data = {
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {
            "temp": 35 + np.random.normal(0, 5),
            "humidity": 45 + np.random.normal(0, 10),
            "pressure": 1013
        },
        "wind": {
            "speed": 5.5 + np.random.normal(0, 2),
            "deg": 180
        },
        "clouds": {"all": 10},
        "sys": {"sunrise": 1640844000, "sunset": 1640883600},
        "tmy_dataframe": None
    }
    return sample_data

# IMPROVED CALCULATION FUNCTIONS
def calculate_solar_position(lat, day_of_year, hour):
    """Calculate solar position angles for improved accuracy."""
    lat_rad = math.radians(lat)
    # Solar declination angle
    declination = math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    # Hour angle (solar noon = 0)
    hour_angle = math.radians(15 * (hour - 12))
    # Solar elevation angle
    elevation = math.asin(
        math.sin(declination) * math.sin(lat_rad) + 
        math.cos(declination) * math.cos(lat_rad) * math.cos(hour_angle)
    )
    # Solar azimuth angle
    azimuth = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat_rad) - math.tan(declination) * math.cos(lat_rad)
    )
    return math.degrees(elevation), math.degrees(azimuth)

def calculate_pv_output_factor(elevation_angle, azimuth_angle, panel_tilt, panel_azimuth):
    """Calculate PV output factor based on panel orientation and sun position."""
    if elevation_angle <= 0:
        return 0
    # Convert to radians
    elev_rad = math.radians(elevation_angle)
    azim_rad = math.radians(azimuth_angle)
    tilt_rad = math.radians(panel_tilt)
    panel_azim_rad = math.radians(panel_azimuth)
    # Calculate incidence angle on tilted panel
    cos_incidence = (
        math.sin(elev_rad) * math.cos(tilt_rad) +
        math.cos(elev_rad) * math.sin(tilt_rad) * math.cos(azim_rad - panel_azim_rad)
    )
    # Ensure non-negative
    cos_incidence = max(0, cos_incidence)
    return cos_incidence

def calculate_air_density(temperature_c, pressure_pa, altitude_m=0):
    """Calculate air density for wind power corrections."""
    # Temperature in Kelvin
    temp_k = temperature_c + 273.15
    # Pressure correction for altitude
    pressure_corrected = pressure_pa * (1 - 0.0065 * altitude_m / temp_k) ** 5.255
    # Air density (kg/m³)
    rho = pressure_corrected / (287.05 * temp_k)
    return rho

def wind_power_curve_realistic(wind_speed, turbine_rated_kw, cut_in=3, rated_speed=12, cut_out=25):
    """Realistic wind turbine power curve with better modeling."""
    if wind_speed < cut_in or wind_speed > cut_out:
        return 0
    elif wind_speed <= rated_speed:
        # More realistic power curve using a polynomial fit
        normalized_speed = (wind_speed - cut_in) / (rated_speed - cut_in)
        # Cubic relationship with some smoothing
        power_ratio = -2 * normalized_speed**3 + 3 * normalized_speed**2
        return turbine_rated_kw * power_ratio
    else:
        return turbine_rated_kw

def calculate_enhanced_solar_energy(tmy_df, lat, area_sqm, panel_efficiency, 
                                   system_losses, panel_tilt=25, panel_azimuth=180,
                                   soiling_losses=0.02, shading_losses=0.0):
    """Enhanced solar energy calculation with realistic physics."""
    if tmy_df is None:
        return None
    hourly_energy_kwh = []
    for idx, row in tmy_df.iterrows():
        # Get solar irradiance
        ghi = row['G(h)']  # Global Horizontal Irradiance (W/m²)
        temperature = row['T2m']
        # Calculate solar position
        day_of_year = idx.timetuple().tm_yday
        hour = idx.hour
        elevation, azimuth = calculate_solar_position(lat, day_of_year, hour)
        # Calculate PV output factor for panel orientation
        pv_factor = calculate_pv_output_factor(elevation, azimuth, panel_tilt, panel_azimuth)
        # Effective irradiance on panel
        effective_irradiance = ghi * pv_factor
        # Temperature coefficient (more accurate for silicon panels)
        temp_coeff_per_c = -0.0045  # -0.45%/°C (typical for silicon)
        temp_factor = 1 + temp_coeff_per_c * (temperature - 25)
        # Calculate power output (kW)
        power_kw = (effective_irradiance / 1000) * area_sqm * (panel_efficiency / 100) * temp_factor
        # Apply all losses
        total_losses = system_losses + soiling_losses + shading_losses
        actual_power_kw = power_kw * (1 - total_losses)
        # Energy for this hour (kWh)
        energy_kwh = max(0, actual_power_kw)
        hourly_energy_kwh.append(energy_kwh)
    # Add to dataframe
    tmy_df_copy = tmy_df.copy()
    tmy_df_copy['Solar_Energy_kWh'] = hourly_energy_kwh
    annual_energy_kwh = sum(hourly_energy_kwh)
    daily_energy_kwh = annual_energy_kwh / 365
    monthly_energy_kwh = annual_energy_kwh / 12
    # Calculate additional metrics
    peak_power_kw = area_sqm * (panel_efficiency / 100)  # Peak power under STC
    performance_ratio = annual_energy_kwh / (peak_power_kw * tmy_df['G(h)'].sum() / 1000) if tmy_df['G(h)'].sum() > 0 else 0
    return {
        "daily_kwh": daily_energy_kwh,
        "monthly_kwh": monthly_energy_kwh,
        "annual_kwh": annual_energy_kwh,
        "capacity_kw": peak_power_kw,
        "performance_ratio": performance_ratio,
        "co2_saved_tons": annual_energy_kwh * 0.65,  # More realistic for Gulf region grid
        "avg_temperature": tmy_df['T2m'].mean(),
        "total_irradiance_kwh_m2": tmy_df['G(h)'].sum() / 1000,
        "hourly_data": tmy_df_copy
    }

def calculate_enhanced_wind_energy(tmy_df, turbine_kw, hub_height=80, 
                                  system_losses=0.15, altitude_m=0):
    """Enhanced wind energy calculation with realistic corrections."""
    if tmy_df is None:
        return None
    hourly_energy_kwh = []
    for idx, row in tmy_df.iterrows():
        # Get meteorological data
        wind_speed_10m = row['WS10m']  # Wind speed at 10m
        temperature = row['T2m']
        pressure = row['SP']  # Surface pressure in Pa
        # Wind shear correction to hub height (typically 80-100m for modern turbines)
        alpha = 0.14  # Wind shear exponent (typical for open terrain)
        wind_speed_hub = wind_speed_10m * (hub_height / 10) ** alpha
        # Air density correction
        air_density = calculate_air_density(temperature, pressure, altitude_m)
        air_density_ratio = air_density / 1.225  # Standard air density at sea level
        # Calculate power output using realistic power curve
        base_power_kw = wind_power_curve_realistic(wind_speed_hub, turbine_kw)
        # Apply air density correction (power proportional to air density)
        corrected_power_kw = base_power_kw * air_density_ratio
        # Apply system losses
        actual_power_kw = corrected_power_kw * (1 - system_losses)
        # Energy for this hour (kWh)
        energy_kwh = max(0, actual_power_kw)
        hourly_energy_kwh.append(energy_kwh)
    # Add to dataframe
    tmy_df_copy = tmy_df.copy()
    tmy_df_copy['Wind_Energy_kWh'] = hourly_energy_kwh
    tmy_df_copy['Wind_Speed_Hub'] = tmy_df['WS10m'] * (hub_height / 10) ** 0.14
    annual_energy_kwh = sum(hourly_energy_kwh)
    daily_energy_kwh = annual_energy_kwh / 365
    monthly_energy_kwh = annual_energy_kwh / 12
    # Calculate capacity factor
    capacity_factor = annual_energy_kwh / (turbine_kw * 8760)
    # Average wind speed at hub height
    avg_wind_speed_hub = tmy_df_copy['Wind_Speed_Hub'].mean()
    return {
        "daily_kwh": daily_energy_kwh,
        "monthly_kwh": monthly_energy_kwh,
        "annual_kwh": annual_energy_kwh,
        "capacity_kw": turbine_kw,
        "capacity_factor_actual": capacity_factor,
        "co2_saved_tons": annual_energy_kwh * 0.65,  # More realistic for Gulf region grid
        "avg_wind_speed_10m": tmy_df['WS10m'].mean(),
        "avg_wind_speed_hub": avg_wind_speed_hub,
        "hourly_data": tmy_df_copy
    }

def calculate_enhanced_economics(energy_data, system_type, capacity, 
                               capex_per_unit, opex_percentage=0.02, 
                               electricity_rate=0.08, discount_rate=0.06, 
                               project_lifetime=25, degradation_rate=0.005):
    """Enhanced economic analysis with lifecycle costs."""
    if not energy_data:
        return {}
    # Initial investment
    if system_type == "solar":
        capex = capacity * capex_per_unit  # $/kW for solar
        # Add inverter replacement cost (every 10-12 years)
        inverter_replacement_cost = capacity * 200  # $200/kW for inverter
    else:  # wind
        capex = capacity * capex_per_unit  # $/kW for wind
        inverter_replacement_cost = 0  # Wind turbines have integrated power electronics
    # Annual O&M costs
    annual_opex = capex * opex_percentage
    # Calculate NPV and IRR
    cash_flows = [-capex]  # Initial investment (negative)
    for year in range(1, project_lifetime + 1):
        # Annual energy production with degradation
        annual_energy = energy_data['annual_kwh'] * (1 - degradation_rate) ** (year - 1)
        # Annual revenue
        annual_revenue = annual_energy * electricity_rate
        # Annual costs
        annual_costs = annual_opex
        # Inverter replacement (for solar, every 12 years)
        if system_type == "solar" and year % 12 == 0:
            annual_costs += inverter_replacement_cost
        # Net cash flow
        net_cash_flow = annual_revenue - annual_costs
        cash_flows.append(net_cash_flow)
    # Calculate NPV
    npv = sum([cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows)])
    # Calculate simple payback period
    cumulative_cash_flow = -capex
    payback_years = project_lifetime  # Default to project lifetime if no payback
    for year in range(1, project_lifetime + 1):
        annual_energy = energy_data['annual_kwh'] * (1 - degradation_rate) ** (year - 1)
        annual_net_cash_flow = annual_energy * electricity_rate - annual_opex
        cumulative_cash_flow += annual_net_cash_flow
        if cumulative_cash_flow >= 0 and payback_years == project_lifetime:
            payback_years = year
            break
    # LCOE calculation
    discounted_energy = sum([energy_data['annual_kwh'] * (1 - degradation_rate) ** (year - 1) / (1 + discount_rate) ** year 
                           for year in range(1, project_lifetime + 1)])
    discounted_costs = capex + sum([(annual_opex + (inverter_replacement_cost if system_type == "solar" and year % 12 == 0 else 0)) / (1 + discount_rate) ** year 
                                  for year in range(1, project_lifetime + 1)])
    lcoe = discounted_costs / discounted_energy if discounted_energy > 0 else float('inf')
    return {
        "capex": capex,
        "annual_opex": annual_opex,
        "npv": npv,
        "payback_years": payback_years,
        "lcoe": lcoe,
        "total_lifetime_energy": sum([energy_data['annual_kwh'] * (1 - degradation_rate) ** (year - 1) for year in range(1, project_lifetime + 1)]),
        "irr": None  # Would need iterative calculation for IRR
    }

def create_map_interface():


    """Create professional interactive map with enhanced features."""
    center_lat, center_lon = 23.5, 45.0
    bounds = [[8, 38], [40, 58]]  # [[south, west], [north, east]]
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        min_zoom=5,           # Prevent zooming out
        max_bounds=True,      # Enable bounds restriction
        tiles='CartoDB positron',
        attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
    )

    # Add different map layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='© Esri',
        name='Satellite',
        overlay=False,
        control=True,
        zoom_start=5,
        min_zoom=5,           # Prevent zooming out
        max_bounds=True    # Enable bounds restriction
    ).add_to(m)
    
    # Create marker clusters for better organization
    marker_cluster = MarkerCluster(name='Cities').add_to(m)
        # Apply the bounds to the map
    m.fit_bounds(bounds)
    
    
    # Add cities with enhanced information
    for country, cities in ARABIAN_CITIES.items():
        for city_name, coords in cities.items():
            # Enhanced popup with more information
            popup_html = f"""
            <div style="width: 200px;">
                <h4>{city_name}</h4>
                <p><strong>Country:</strong> {country}</p>
                <p><strong>Coordinates:</strong><br>
                   Lat: {coords['lat']:.4f}°<br>
                   Lon: {coords['lon']:.4f}°</p>
                <p><strong>Elevation:</strong> {coords.get('elevation', 'N/A')} m</p>
            </div>
            """
            
            folium.Marker(
                location=[coords["lat"], coords["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{city_name}, {country}",
                icon=folium.Icon(
                    color='blue' if country == 'Saudi Arabia' else 'green',
                    icon='info-sign'
                )
            ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m


def main():
    st.title("🌞 Enhanced Arabian Peninsula Renewable Energy Assessment")
    st.markdown("### Physics-Based Solar & Wind Energy Analysis with Interactive Map")
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Configuration")
    
    
    # Location selection using map and city selector
    st.sidebar.subheader("📍 Location Selection")
    st.sidebar.markdown("Select a major city or click anywhere on the map:")
    
    # Create a dropdown for predefined cities
    city_options = ["-- Select a City --"]
    city_to_coords = {}
    for country, cities in ARABIAN_CITIES.items():
        for city_name in cities.keys():
            display_name = f"{city_name}, {country}"
            city_options.append(display_name)
            city_to_coords[display_name] = {
                "lat": cities[city_name]["lat"],
                "lon": cities[city_name]["lon"],
                "name": city_name,
                "country": country
            }
    
    selected_city = st.sidebar.selectbox("Choose a City", city_options)
    
    # Create and display map
    st.sidebar.markdown("Or, click on the map to select a custom location:")
    map_data = create_map_interface()
    map_result = st_folium(map_data, width="100%", height=800, key="location_map")
    
    # Determine selected location
    if selected_city != "-- Select a City --":
        # Use the coordinates from the selected city
        city_info = city_to_coords[selected_city]
        selected_location = {
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "name": city_info["name"],
            "country": city_info["country"]
        }
        st.sidebar.success(f"Selected City: {selected_city}")
    elif map_result["last_clicked"] and map_result["last_clicked"]["lat"]:
        # Use coordinates from map click
        selected_location = {
            "lat": map_result["last_clicked"]["lat"],
            "lon": map_result["last_clicked"]["lng"],
            "name": "Custom Location",
            "country": "N/A"
        }
        st.sidebar.success(f"Selected: {selected_location['lat']:.3f}, {selected_location['lon']:.3f}")
    else:
        # Default location
        default_city = "Sanaa, Yemen"
        selected_location = {
            "lat": ARABIAN_CITIES["Yemen"]["Sanaa"]["lat"],
            "lon": ARABIAN_CITIES["Yemen"]["Sanaa"]["lon"],
            "name": "Sanaa",
            "country": "Yemen"
        }
        st.sidebar.info(f"Default location selected: {default_city}")
    
    # Energy type selection
    st.sidebar.subheader("⚡ Energy System Configuration")
    energy_types = st.sidebar.multiselect(
        "Choose Energy Types",
        ["Solar Energy", "Wind Energy"],
        default=["Solar Energy"]
    )
    
    # Enhanced Solar Configuration
    if "Solar Energy" in energy_types:
        st.sidebar.subheader("☀️ Solar System Configuration")
        solar_capacity_kw = st.sidebar.number_input(
            "Solar System Capacity (kW)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )
        # Calculate area from capacity (assuming 400W panels)
        panel_power_rating = st.sidebar.slider("Panel Power Rating (W)", 300, 600, 400, 25)
        solar_area = (solar_capacity_kw * 1000) / panel_power_rating * 2  # Approximate area with spacing
        panel_efficiency = st.sidebar.slider("Panel Efficiency (%)", 15, 25, 21, 1)
        panel_tilt = st.sidebar.slider("Panel Tilt Angle (°)", 0, 60, 25, 5)
        panel_azimuth = st.sidebar.slider("Panel Azimuth (° from South)", -90, 90, 0, 15)
        soiling_losses = st.sidebar.slider("Soiling Losses (%)", 0.0, 10.0, 3.0, 0.5) / 100
        shading_losses = st.sidebar.slider("Shading Losses (%)", 0.0, 20.0, 2.0, 1.0) / 100
    
    # Enhanced Wind Configuration
    if "Wind Energy" in energy_types:
        st.sidebar.subheader("💨 Wind System Configuration")
        wind_capacity = st.sidebar.number_input(
            "Wind Turbine Capacity (kW)",
            min_value=1.0,
            max_value=5000.0,
            value=2500.0,
            step=100.0
        )
        hub_height = st.sidebar.slider("Hub Height (m)", 50, 150, 100, 10)
        altitude = st.sidebar.number_input("Site Altitude (m)", 0, 3000, 0, 100)
    
    # System-wide parameters
    st.sidebar.subheader("⚙️ System Parameters")
    system_losses = st.sidebar.slider("System Losses (%)", 5.0, 25.0, 12.0, 1.0) / 100
    
    # Enhanced Economic Parameters
    st.sidebar.subheader("💰 Economic Parameters")
    if "Solar Energy" in energy_types:
        solar_capex = st.sidebar.number_input(
            "Solar CAPEX ($/kW)",
            min_value=300.0,
            max_value=2500.0,
            value=1200.0,
            step=50.0
        )
    if "Wind Energy" in energy_types:
        wind_capex = st.sidebar.number_input(
            "Wind CAPEX ($/kW)",
            min_value=800.0,
            max_value=2500.0,
            value=1800.0,
            step=50.0
        )
    opex_rate = st.sidebar.slider("O&M Cost (% of CAPEX/year)", 1.0, 5.0, 2.5, 0.5) / 100
    electricity_rate = st.sidebar.number_input(
        "Electricity Rate ($/kWh)",
        min_value=0.02,
        max_value=0.25,
        value=0.08,
        step=0.01,
        format="%.3f"
    )
    discount_rate = st.sidebar.slider("Discount Rate (%)", 3.0, 12.0, 6.0, 0.5) / 100
    project_lifetime = st.sidebar.slider("Project Lifetime (years)", 15, 30, 25, 1)
    
    if st.sidebar.button("🚀 Calculate Enhanced Assessment", type="primary"):
        # Get weather data
        weather_data = get_weather_data(
            selected_location["lat"], 
            selected_location["lon"], 
            city_name=selected_location.get("name", "Unknown"),
            country_code=selected_location.get("country", "").split(' ')[0][:3].upper() # Simple country code extraction
        )
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        with col1:
            st.header(f"📊 Enhanced Energy Assessment")
            st.subheader(f"Location: {selected_location.get('name', 'Custom')}, {selected_location.get('country', '')}")
            st.write(f"Coordinates: {selected_location['lat']:.3f}°N, {selected_location['lon']:.3f}°E")
            
            # Weather information
            st.subheader("🌤️ Current Conditions")
            weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
            with weather_col1:
                st.metric("Temperature", f"{weather_data['main']['temp']:.1f}°C")
            with weather_col2:
                st.metric("Wind Speed", f"{weather_data['wind']['speed']:.1f} m/s")
            with weather_col3:
                st.metric("Humidity", f"{weather_data['main']['humidity']}%")
            with weather_col4:
                st.metric("Pressure", f"{weather_data['main']['pressure']:.0f} hPa")
        with col2:
            st.subheader("📍 Location Details")
            st.write(f"**Coordinates:** {selected_location['lat']:.4f}°, {selected_location['lon']:.4f}°")
            st.write(f"**Weather:** {weather_data['weather'][0]['description'].title()}")
            # Show if using TMY data
            if weather_data.get('tmy_dataframe') is not None:
                st.success("✅ Using detailed historical TMY data")
                tmy_hours = len(weather_data['tmy_dataframe'])
                st.write(f"**Data Points:** {tmy_hours:,} hours")
            else:
                st.info("ℹ️ Using sample/weather API data")
        
        # Calculate enhanced energy production
        solar_results = None
        wind_results = None
        tmy_df = weather_data.get('tmy_dataframe')
        if "Solar Energy" in energy_types:
            if tmy_df is not None:
                solar_results = calculate_enhanced_solar_energy(
                    tmy_df, selected_location['lat'], solar_area, panel_efficiency,
                    system_losses, panel_tilt, panel_azimuth + 180,  # Convert to compass bearing
                    soiling_losses, shading_losses
                )
                st.success(f"☀️ Enhanced solar calculation with {len(tmy_df):,} hours of data")
            else:
                st.warning("☀️ No TMY data available - calculations may be less accurate")
        if "Wind Energy" in energy_types:
            if tmy_df is not None:
                wind_results = calculate_enhanced_wind_energy(
                    tmy_df, wind_capacity, hub_height, system_losses, altitude
                )
                st.success(f"💨 Enhanced wind calculation with {len(tmy_df):,} hours of data")
            else:
                st.warning("💨 No TMY data available - calculations may be less accurate")
        
        # Enhanced economic analysis
        solar_economics = {}
        wind_economics = {}
        if solar_results:
            solar_economics = calculate_enhanced_economics(
                solar_results, "solar", solar_capacity_kw, solar_capex,
                opex_rate, electricity_rate, discount_rate, project_lifetime
            )
        if wind_results:
            wind_economics = calculate_enhanced_economics(
                wind_results, "wind", wind_capacity, wind_capex,
                opex_rate, electricity_rate, discount_rate, project_lifetime
            )
        
        # Display results
        st.header("📈 Enhanced Energy Production Results")
        if solar_results or wind_results:
            # Create enhanced tabs
            tab_titles = ["📊 Production Overview", "📅 Monthly Analysis", 
                         "💰 Enhanced Economics", "🌍 Environmental Impact",
                         "🔬 Technical Performance"]
            # Add Time Series Analysis tab if TMY data available
            if (solar_results and 'hourly_data' in solar_results) or (wind_results and 'hourly_data' in wind_results):
                tab_titles.append("⏱️ Time Series Analysis")
            tabs = st.tabs(tab_titles)
            
            with tabs[0]:  # Production Overview
                if solar_results and wind_results:
                    col1, col2 = st.columns(2)
                else:
                    col1, col2 = st.columns([1, 1])
                
                if solar_results:
                    with col1:
                        st.subheader("☀️ Enhanced Solar Results")
                        st.metric("Daily Production", f"{solar_results['daily_kwh']:.1f} kWh")
                        st.metric("Monthly Production", f"{solar_results['monthly_kwh']:.1f} kWh")
                        st.metric("Annual Production", f"{solar_results['annual_kwh']:.0f} kWh")
                        st.metric("System Capacity", f"{solar_results['capacity_kw']:.1f} kWp")
                        st.metric("Performance Ratio", f"{solar_results.get('performance_ratio', 0):.2f}")
                        st.metric("Solar Resource", f"{solar_results['total_irradiance_kwh_m2']:.0f} kWh/m²")
                        st.metric("Capacity Factor", f"{(solar_results['annual_kwh']/(solar_results['capacity_kw']*8760)):.1%}")
                
                if wind_results:
                    with col2:
                        st.subheader("💨 Enhanced Wind Results")
                        st.metric("Daily Production", f"{wind_results['daily_kwh']:.1f} kWh")
                        st.metric("Monthly Production", f"{wind_results['monthly_kwh']:.1f} kWh")
                        st.metric("Annual Production", f"{wind_results['annual_kwh']:.0f} kWh")
                        st.metric("System Capacity", f"{wind_results['capacity_kw']:.1f} kW")
                        st.metric("Capacity Factor", f"{wind_results.get('capacity_factor_actual', 0):.1%}")
                        st.metric("Avg Wind Speed (10m)", f"{wind_results.get('avg_wind_speed_10m', 0):.1f} m/s")
                        st.metric("Avg Wind Speed (Hub)", f"{wind_results.get('avg_wind_speed_hub', 0):.1f} m/s")
                
                # Enhanced production comparison chart
                # Enhanced production comparison chart
                if solar_results and wind_results:
                    st.subheader("⚡ Energy Production Comparison")
                    comparison_data = {
                        'Technology': ['Solar PV', 'Wind'],
                        'Annual Production (MWh)': [
                            solar_results['annual_kwh'] / 1000,
                            wind_results['annual_kwh'] / 1000
                        ],
                        'Capacity Factor (%)': [
                            (solar_results['annual_kwh']/(solar_results['capacity_kw']*8760)) * 100,
                            wind_results.get('capacity_factor_actual', 0) * 100
                        ],
                        'Installed Capacity (MW)': [
                            solar_results['capacity_kw'] / 1000,
                            wind_results['capacity_kw'] / 1000
                        ]
                    }
                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=('Annual Production (MWh)', 'Capacity Factor (%)', 'Installed Capacity (MW)')
                    )
                    colors = ['orange', 'lightblue']
                    # Use the keys from comparison_data directly to avoid typos
                    metrics = ['Annual Production (MWh)', 'Capacity Factor (%)', 'Installed Capacity (MW)']
                    for i, metric in enumerate(metrics):
                        fig.add_trace(
                            go.Bar(x=comparison_data['Technology'], 
                                y=comparison_data[metric],
                                name=metric,
                                marker_color=colors,
                                showlegend=False),
                            row=1, col=i+1
                        )
                    fig.update_layout(height=400, title_text="Technology Performance Comparison")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:  # Monthly Analysis
                st.subheader("📅 Monthly Production Analysis")
                if (solar_results and 'hourly_data' in solar_results) or (wind_results and 'hourly_data' in wind_results):
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    fig = go.Figure()
                    if solar_results and 'hourly_data' in solar_results:
                        solar_df = solar_results['hourly_data']
                        monthly_solar = []
                        for month in range(1, 13):
                            month_data = solar_df[solar_df.index.month == month]['Solar_Energy_kWh'].sum()
                            monthly_solar.append(month_data)
                        fig.add_trace(go.Scatter(
                            x=months,
                            y=monthly_solar,
                            mode='lines+markers',
                            name='Solar Energy',
                            line=dict(color='orange', width=3),
                            marker=dict(size=8)
                        ))
                    if wind_results and 'hourly_data' in wind_results:
                        wind_df = wind_results['hourly_data']
                        monthly_wind = []
                        for month in range(1, 13):
                            month_data = wind_df[wind_df.index.month == month]['Wind_Energy_kWh'].sum()
                            monthly_wind.append(month_data)
                        fig.add_trace(go.Scatter(
                            x=months,
                            y=monthly_wind,
                            mode='lines+markers',
                            name='Wind Energy',
                            line=dict(color='lightblue', width=3),
                            marker=dict(size=8)
                        ))
                    fig.update_layout(
                        title='Monthly Energy Production Profile',
                        xaxis_title='Month',
                        yaxis_title='Energy Production (kWh)',
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.info("Monthly variations are calculated using actual TMY data accounting for seasonal weather patterns.")
            
            with tabs[2]:  # Enhanced Economics
                st.subheader("💰 Comprehensive Economic Analysis")
                if solar_economics or wind_economics:
                    # Financial metrics comparison
                    financial_data = []
                    if solar_economics:
                        financial_data.append({
                            'Technology': 'Solar PV',
                            'CAPEX ($)': f"{solar_economics['capex']:,.0f}",
                            'Annual O&M ($)': f"{solar_economics['annual_opex']:,.0f}",
                            'NPV ($)': f"{solar_economics['npv']:,.0f}",
                            'LCOE ($/kWh)': f"{solar_economics['lcoe']:.3f}",
                            'Payback (years)': f"{solar_economics['payback_years']:.1f}",
                            'Lifetime Energy (MWh)': f"{solar_economics['total_lifetime_energy']/1000:.0f}"
                        })
                    if wind_economics:
                        financial_data.append({
                            'Technology': 'Wind',
                            'CAPEX ($)': f"{wind_economics['capex']:,.0f}",
                            'Annual O&M ($)': f"{wind_economics['annual_opex']:,.0f}",
                            'NPV ($)': f"{wind_economics['npv']:,.0f}",
                            'LCOE ($/kWh)': f"{wind_economics['lcoe']:.3f}",
                            'Payback (years)': f"{wind_economics['payback_years']:.1f}",
                            'Lifetime Energy (MWh)': f"{wind_economics['total_lifetime_energy']/1000:.0f}"
                        })
                    financial_df = pd.DataFrame(financial_data)
                    st.dataframe(financial_df, use_container_width=True)
                    
                    # LCOE comparison chart
                    if len(financial_data) > 1:
                        lcoe_data = []
                        for data in financial_data:
                            lcoe_data.append({
                                'Technology': data['Technology'],
                                'LCOE': float(data['LCOE ($/kWh)'])
                            })
                        lcoe_df = pd.DataFrame(lcoe_data)
                        fig = go.Figure(data=[
                            go.Bar(x=lcoe_df['Technology'], 
                                  y=lcoe_df['LCOE'],
                                  marker_color=['orange', 'lightblue'],
                                  text=[f"${x:.3f}" for x in lcoe_df['LCOE']],
                                  textposition='auto')
                        ])
                        fig.update_layout(
                            title='Levelized Cost of Energy (LCOE) Comparison',
                            xaxis_title='Technology',
                            yaxis_title='LCOE ($/kWh)',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cash flow analysis
                    st.subheader("📊 25-Year Cash Flow Analysis")
                    if solar_economics and wind_economics:
                        years = list(range(0, 26))
                        solar_cashflow = [-solar_economics['capex']]
                        wind_cashflow = [-wind_economics['capex']]
                        for year in range(1, 26):
                            solar_annual = solar_results['annual_kwh'] * (1 - 0.005) ** (year-1) * electricity_rate - solar_economics['annual_opex']
                            wind_annual = wind_results['annual_kwh'] * (1 - 0.005) ** (year-1) * electricity_rate - wind_economics['annual_opex']
                            solar_cashflow.append(solar_cashflow[-1] + solar_annual)
                            wind_cashflow.append(wind_cashflow[-1] + wind_annual)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=years, y=solar_cashflow,
                            mode='lines+markers',
                            name='Solar PV',
                            line=dict(color='orange', width=3)
                        ))
                        fig.add_trace(go.Scatter(
                            x=years, y=wind_cashflow,
                            mode='lines+markers',
                            name='Wind',
                            line=dict(color='lightblue', width=3)
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                                     annotation_text="Break-even line")
                        fig.update_layout(
                            title='Cumulative Cash Flow Over Project Lifetime',
                            xaxis_title='Year',
                            yaxis_title='Cumulative Cash Flow ($)',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.info("💡 Economic analysis includes system degradation, O&M costs, and NPV calculations at specified discount rate.")
            
            with tabs[3]:  # Environmental Impact
                st.subheader("🌍 Enhanced Environmental Analysis")
                total_co2_saved = 0
                total_annual_production = 0
                if solar_results:
                    total_co2_saved += solar_results['co2_saved_tons']
                    total_annual_production += solar_results['annual_kwh']
                if wind_results:
                    total_co2_saved += wind_results['co2_saved_tons']
                    total_annual_production += wind_results['annual_kwh']
                
                # Enhanced environmental metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("CO₂ Saved/Year", f"{total_co2_saved:.1f} tons")
                with col2:
                    st.metric("CO₂ Intensity", f"{total_co2_saved/total_annual_production*1000:.0f} g/kWh")
                with col3:
                    st.metric("Tree Equivalent", f"{int(total_co2_saved * 16)} trees/year")
                with col4:
                    st.metric("Homes Powered", f"{int(total_annual_production / 10000)} homes")
                
                # Lifecycle environmental impact
                years = list(range(1, 26))
                cumulative_co2 = [total_co2_saved * year for year in years]
                # Account for manufacturing emissions (solar: 40g CO2/kWh, wind: 15g CO2/kWh over lifetime)
                manufacturing_emissions = 0
                if solar_results:
                    manufacturing_emissions += solar_results['annual_kwh'] * 25 * 0.04 / 1000  # tons
                if wind_results:
                    manufacturing_emissions += wind_results['annual_kwh'] * 25 * 0.015 / 1000  # tons
                net_co2_savings = [co2 - manufacturing_emissions for co2 in cumulative_co2]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, y=cumulative_co2,
                    mode='lines+markers',
                    name='Gross CO₂ Savings',
                    line=dict(color='green', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=years, y=net_co2_savings,
                    mode='lines+markers',
                    name='Net CO₂ Savings (after manufacturing)',
                    line=dict(color='darkgreen', width=3, dash='dot')
                ))
                fig.update_layout(
                    title='Lifecycle CO₂ Impact Analysis',
                    xaxis_title='Years',
                    yaxis_title='CO₂ Savings (tons)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"🎉 Net CO₂ savings over 25 years: {net_co2_savings[-1]:.0f} tons")
                st.info("Analysis includes manufacturing emissions and grid displacement factors for the Gulf region.")
            
            with tabs[4]:  # Technical Performance
                st.subheader("🔬 Technical Performance Analysis")
                if solar_results:
                    st.subheader("☀️ Solar Technical Metrics")
                    solar_col1, solar_col2, solar_col3 = st.columns(3)
                    with solar_col1:
                        st.metric("Performance Ratio", f"{solar_results.get('performance_ratio', 0):.2f}")
                        st.metric("System Efficiency", f"{panel_efficiency:.1f}%")
                        st.metric("Panel Tilt", f"{panel_tilt}°")
                    with solar_col2:
                        st.metric("Array Area", f"{solar_area:.0f} m²")
                        st.metric("Specific Yield", f"{solar_results['annual_kwh']/solar_results['capacity_kw']:.0f} kWh/kWp")
                        st.metric("Panel Azimuth", f"{panel_azimuth}°")
                    with solar_col3:
                        st.metric("Total Losses", f"{(system_losses + soiling_losses + shading_losses)*100:.1f}%")
                        st.metric("Temperature Deration", f"{((solar_results['avg_temperature'] - 25) * -0.45):.1f}%")
                        st.metric("Solar Resource", f"{solar_results['total_irradiance_kwh_m2']:.0f} kWh/m²")
                
                if wind_results:
                    st.subheader("💨 Wind Technical Metrics")
                    wind_col1, wind_col2, wind_col3 = st.columns(3)
                    with wind_col1:
                        st.metric("Hub Height", f"{hub_height} m")
                        st.metric("Wind Speed (10m)", f"{wind_results.get('avg_wind_speed_10m', 0):.1f} m/s")
                        st.metric("Wind Speed (Hub)", f"{wind_results.get('avg_wind_speed_hub', 0):.1f} m/s")
                    with wind_col2:
                        st.metric("Capacity Factor", f"{wind_results.get('capacity_factor_actual', 0):.1%}")
                        st.metric("System Losses", f"{system_losses*100:.1f}%")
                        st.metric("Site Altitude", f"{altitude} m")
                    with wind_col3:
                        annual_hours = 8760
                        equiv_hours = wind_results['annual_kwh'] / wind_results['capacity_kw']
                        st.metric("Equivalent Hours", f"{equiv_hours:.0f} hours")
                        st.metric("Availability", "97%")  # Typical modern turbine
                        st.metric("Wind Shear Factor", "0.14")
                
                # Power curves visualization
                if wind_results:
                    st.subheader("💨 Wind Turbine Power Curve")
                    wind_speeds = np.arange(0, 30, 0.5)
                    power_outputs = [wind_power_curve_realistic(ws, wind_capacity) for ws in wind_speeds]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wind_speeds,
                        y=power_outputs,
                        mode='lines',
                        name='Power Curve',
                        line=dict(color='blue', width=3)
                    ))
                    # Add actual wind speed line
                    if wind_results.get('avg_wind_speed_hub'):
                        fig.add_vline(
                            x=wind_results['avg_wind_speed_hub'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Site Avg: {wind_results['avg_wind_speed_hub']:.1f} m/s"
                        )
                    fig.update_layout(
                        title=f'Wind Turbine Power Curve ({wind_capacity} kW)',
                        xaxis_title='Wind Speed (m/s)',
                        yaxis_title='Power Output (kW)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time Series Analysis tab (if TMY data available)
            if len(tabs) > 5:
                with tabs[5]:  # Time Series Analysis
                    st.subheader("⏱️ Detailed Time Series Analysis")
                    
                    # Analysis configuration
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        time_period = st.selectbox(
                            "Select Analysis Period",
                            ["Daily (24 hours)", "Weekly (7 days)", "Monthly (30 days)", "Seasonal (1 year)"],
                            index=0
                        )
                    
                    with analysis_col2:
                        start_date = st.selectbox(
                            "Starting Period",
                            ["January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"],
                            index=0
                        )
                    
                    # Generate time series plots based on TMY data
                    if solar_results and 'hourly_data' in solar_results:
                        st.subheader("☀️ Solar System Time Series Analysis")
                        
                        df = solar_results['hourly_data']
                        
                        # Get starting month data
                        start_month = ["January", "February", "March", "April", "May", "June",
                                      "July", "August", "September", "October", "November", "December"].index(start_date) + 1
                        
                        # Filter data based on period and starting month
                        if time_period == "Daily (24 hours)":
                            # Get first day of selected month
                            month_data = df[df.index.month == start_month]
                            if len(month_data) >= 24:
                                plot_data = month_data.head(24)
                                x_col = plot_data.index.strftime('%H:%M')
                                title_suffix = f"Daily Pattern - {start_date}"
                            else:
                                plot_data = df.head(24)
                                x_col = plot_data.index.strftime('%H:%M')
                                title_suffix = "Daily Pattern - Sample Day"
                                
                        elif time_period == "Weekly (7 days)":
                            month_data = df[df.index.month == start_month]
                            if len(month_data) >= 168:
                                plot_data = month_data.head(168)
                                x_col = plot_data.index.strftime('%d %H:%M')
                                title_suffix = f"Weekly Pattern - {start_date}"
                            else:
                                plot_data = df.head(168)
                                x_col = plot_data.index.strftime('%m-%d %H:%M')
                                title_suffix = "Weekly Pattern - Sample Week"
                                
                        elif time_period == "Monthly (30 days)":
                            month_data = df[df.index.month == start_month]
                            if len(month_data) >= 720:
                                plot_data = month_data.head(720)
                                x_col = plot_data.index.strftime('%d-%H')
                                title_suffix = f"Monthly Pattern - {start_date}"
                            else:
                                plot_data = df.head(720)
                                x_col = plot_data.index.strftime('%m-%d')
                                title_suffix = "Monthly Pattern - Sample Month"
                                
                        else:  # Seasonal (1 year)
                            plot_data = df
                            x_col = plot_data.index.strftime('%m-%d')
                            title_suffix = "Full Year Analysis"
                        
                        # Create enhanced subplots for solar
                        fig = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=(
                                'Solar Irradiance and Energy Production',
                                'Temperature Effects and Performance Ratio',
                                'System Efficiency and Loss Analysis'
                            ),
                            vertical_spacing=0.08,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
                        )
                        
                        # Plot 1: Solar irradiance and energy production
                        fig.add_trace(
                            go.Scatter(x=x_col, y=plot_data['G(h)'][:len(x_col)], 
                                     name="Global Horizontal Irradiance", 
                                     line=dict(color='orange', width=2)),
                            row=1, col=1, secondary_y=False
                        )
                        
                        if 'Gb(n)' in plot_data.columns:
                            fig.add_trace(
                                go.Scatter(x=x_col, y=plot_data['Gb(n)'][:len(x_col)], 
                                         name="Direct Normal Irradiance", 
                                         line=dict(color='gold', width=1, dash='dot')),
                                row=1, col=1, secondary_y=False
                            )
                        
                        fig.add_trace(
                            go.Scatter(x=x_col, y=plot_data['Solar_Energy_kWh'][:len(x_col)], 
                                     name="Energy Production", 
                                     line=dict(color='red', width=3)),
                            row=1, col=1, secondary_y=True
                        )
                        
                        # Plot 2: Temperature effects and performance ratio
                        fig.add_trace(
                            go.Scatter(x=x_col, y=plot_data['T2m'][:len(x_col)], 
                                     name="Ambient Temperature", 
                                     line=dict(color='green', width=2)),
                            row=2, col=1, secondary_y=False
                        )
                        
                        # Calculate cell temperature (simplified model)
                        cell_temp = plot_data['T2m'] + (plot_data['G(h)'] / 1000) * 20  # Approx 20°C rise per kW/m²
                        fig.add_trace(
                            go.Scatter(x=x_col, y=cell_temp[:len(x_col)], 
                                     name="Estimated Cell Temperature", 
                                     line=dict(color='darkgreen', width=1, dash='dash')),
                            row=2, col=1, secondary_y=False
                        )
                        
                        # Calculate instantaneous performance ratio
                        inst_pr = []
                        temp_derate = []
                        for _, row in plot_data.iterrows():
                            if row['G(h)'] > 50:  # Only calculate when sufficient irradiance
                                theoretical_power = (row['G(h)'] / 1000) * solar_results['capacity_kw']
                                actual_power = row['Solar_Energy_kWh']
                                pr = actual_power / theoretical_power if theoretical_power > 0 else 0
                                
                                # Temperature derating factor
                                cell_t = row['T2m'] + (row['G(h)'] / 1000) * 20
                                temp_factor = 1 + (-0.0045) * (cell_t - 25)
                                temp_derate.append(temp_factor)
                            else:
                                pr = 0
                                temp_derate.append(1.0)
                            inst_pr.append(min(pr, 1.0))  # Cap at 1.0
                        
                        fig.add_trace(
                            go.Scatter(x=x_col, y=inst_pr[:len(x_col)], 
                                     name="Performance Ratio", 
                                     line=dict(color='blue', width=2)),
                            row=2, col=1, secondary_y=True
                        )
                        
                        # Plot 3: System efficiency and losses
                        fig.add_trace(
                            go.Scatter(x=x_col, y=temp_derate[:len(x_col)], 
                                     name="Temperature Derating Factor", 
                                     line=dict(color='purple', width=2)),
                            row=3, col=1, secondary_y=False
                        )
                        
                        # Calculate instantaneous system efficiency
                        system_eff = []
                        for _, row in plot_data.iterrows():
                            if row['G(h)'] > 50:
                                efficiency = (row['Solar_Energy_kWh'] * 1000) / (row['G(h)'] * solar_area) if row['G(h)'] > 0 else 0
                            else:
                                efficiency = 0
                            system_eff.append(min(efficiency * 100, panel_efficiency))  # Cap at panel efficiency
                        
                        fig.add_trace(
                            go.Scatter(x=x_col, y=system_eff[:len(x_col)], 
                                     name="Instantaneous System Efficiency", 
                                     line=dict(color='magenta', width=2)),
                            row=3, col=1, secondary_y=True
                        )
                        
                        # Add humidity impact (for soiling estimation)
                        if 'RH' in plot_data.columns:
                            soiling_factor = 1 - (soiling_losses * (100 - plot_data['RH']) / 100)  # Worse soiling in dry conditions
                            fig.add_trace(
                                go.Scatter(x=x_col, y=soiling_factor[:len(x_col)], 
                                         name="Soiling Impact Factor", 
                                         line=dict(color='brown', width=1, dash='dot')),
                                row=3, col=1, secondary_y=False
                            )
                        
                        # Update y-axis labels
                        fig.update_yaxes(title_text="Irradiance (W/m²)", row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="Energy Production (kWh)", row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="Performance Ratio", row=2, col=1, secondary_y=True, range=[0, 1])
                        fig.update_yaxes(title_text="Derating Factor", row=3, col=1, secondary_y=False, range=[0.5, 1.1])
                        fig.update_yaxes(title_text="System Efficiency (%)", row=3, col=1, secondary_y=True)
                        
                        fig.update_layout(
                            height=900, 
                            title_text=f"Solar System Performance Analysis - {title_suffix}",
                            showlegend=True
                        )
                        
                        # Rotate x-axis labels for better readability
                        fig.update_xaxes(tickangle=45)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add statistical summary for solar
                        st.subheader("📊 Solar Performance Statistics")
                        
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        
                        with stats_col1:
                            st.metric("Avg Solar Resource", f"{plot_data['G(h)'].mean():.0f} W/m²")
                            st.metric("Peak Irradiance", f"{plot_data['G(h)'].max():.0f} W/m²")
                        
                        with stats_col2:
                            st.metric("Avg Energy Output", f"{plot_data['Solar_Energy_kWh'].mean():.2f} kWh")
                            st.metric("Peak Energy Output", f"{plot_data['Solar_Energy_kWh'].max():.2f} kWh")
                        
                        with stats_col3:
                            st.metric("Avg Performance Ratio", f"{np.mean(inst_pr):.3f}")
                            st.metric("Avg Temperature", f"{plot_data['T2m'].mean():.1f}°C")
                        
                        with stats_col4:
                            st.metric("Energy Variance", f"{plot_data['Solar_Energy_kWh'].std():.2f} kWh")
                            st.metric("Temperature Range", f"{plot_data['T2m'].max() - plot_data['T2m'].min():.1f}°C")
                    
                    # Enhanced Wind Analysis
                    if wind_results and 'hourly_data' in wind_results:
                        st.subheader("💨 Wind System Time Series Analysis")
                        
                        wind_df = wind_results['hourly_data']
                        
                        # Apply same time filtering logic for wind data
                        if time_period == "Daily (24 hours)":
                            month_data = wind_df[wind_df.index.month == start_month]
                            if len(month_data) >= 24:
                                wind_plot_data = month_data.head(24)
                                wind_x_col = wind_plot_data.index.strftime('%H:%M')
                                wind_title_suffix = f"Daily Wind Pattern - {start_date}"
                            else:
                                wind_plot_data = wind_df.head(24)
                                wind_x_col = wind_plot_data.index.strftime('%H:%M')
                                wind_title_suffix = "Daily Wind Pattern - Sample Day"
                                
                        elif time_period == "Weekly (7 days)":
                            month_data = wind_df[wind_df.index.month == start_month]
                            if len(month_data) >= 168:
                                wind_plot_data = month_data.head(168)
                                wind_x_col = wind_plot_data.index.strftime('%d %H:%M')
                                wind_title_suffix = f"Weekly Wind Pattern - {start_date}"
                            else:
                                wind_plot_data = wind_df.head(168)
                                wind_x_col = wind_plot_data.index.strftime('%m-%d %H:%M')
                                wind_title_suffix = "Weekly Wind Pattern - Sample Week"
                                
                        elif time_period == "Monthly (30 days)":
                            month_data = wind_df[wind_df.index.month == start_month]
                            if len(month_data) >= 720:
                                wind_plot_data = month_data.head(720)
                                wind_x_col = wind_plot_data.index.strftime('%d-%H')
                                wind_title_suffix = f"Monthly Wind Pattern - {start_date}"
                            else:
                                wind_plot_data = wind_df.head(720)
                                wind_x_col = wind_plot_data.index.strftime('%m-%d')
                                wind_title_suffix = "Monthly Wind Pattern - Sample Month"
                                
                        else:  # Seasonal
                            wind_plot_data = wind_df
                            wind_x_col = wind_plot_data.index.strftime('%m-%d')
                            wind_title_suffix = "Full Year Wind Analysis"
                        
                        # Create enhanced wind subplots
                        wind_fig = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=(
                                'Wind Speed Profile and Energy Production',
                                'Air Density Effects and Turbulence Analysis',
                                'Power Curve Performance and Capacity Factor'
                            ),
                            vertical_spacing=0.08,
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
                        )
                        
                        # Plot 1: Wind speed and energy production
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=wind_plot_data['WS10m'][:len(wind_x_col)], 
                                     name="Wind Speed (10m)", 
                                     line=dict(color='blue', width=2)),
                            row=1, col=1, secondary_y=False
                        )
                        
                        # Hub height wind speed
                        if 'Wind_Speed_Hub' in wind_plot_data.columns:
                            wind_fig.add_trace(
                                go.Scatter(x=wind_x_col, y=wind_plot_data['Wind_Speed_Hub'][:len(wind_x_col)], 
                                         name=f"Wind Speed ({hub_height}m hub)", 
                                         line=dict(color='darkblue', width=2, dash='dot')),
                                row=1, col=1, secondary_y=False
                            )
                        
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=wind_plot_data['Wind_Energy_kWh'][:len(wind_x_col)], 
                                     name="Energy Production", 
                                     line=dict(color='lightblue', width=3)),
                            row=1, col=1, secondary_y=True
                        )
                        
                        # Plot 2: Air density and turbulence effects
                        # Calculate air density for each hour
                        air_densities = []
                        for _, row in wind_plot_data.iterrows():
                            rho = calculate_air_density(row['T2m'], row['SP'], altitude)
                            air_densities.append(rho)
                        
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=air_densities[:len(wind_x_col)], 
                                     name="Air Density", 
                                     line=dict(color='green', width=2)),
                            row=2, col=1, secondary_y=False
                        )
                        
                        # Temperature for reference
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=wind_plot_data['T2m'][:len(wind_x_col)], 
                                     name="Temperature", 
                                     line=dict(color='red', width=1, dash='dash')),
                            row=2, col=1, secondary_y=True
                        )
                        
                        # Wind direction variability (as turbulence indicator)
                        if len(wind_plot_data) > 1:
                            wind_dir_change = abs(wind_plot_data['WD10m'].diff()).fillna(0)
                            # Normalize direction changes (account for 360° wrap-around)
                            wind_dir_change = wind_dir_change.apply(lambda x: min(x, 360-x) if x > 180 else x)
                            
                            wind_fig.add_trace(
                                go.Scatter(x=wind_x_col, y=wind_dir_change[:len(wind_x_col)], 
                                         name="Wind Direction Variability", 
                                         line=dict(color='orange', width=1)),
                                row=2, col=1, secondary_y=True
                            )
                        
                        # Plot 3: Power curve performance
                        # Calculate theoretical vs actual power
                        theoretical_power = []
                        actual_power = wind_plot_data['Wind_Energy_kWh'][:len(wind_x_col)].tolist()
                        capacity_factor_inst = []
                        
                        for _, row in wind_plot_data.head(len(wind_x_col)).iterrows():
                            hub_speed = row.get('Wind_Speed_Hub', row['WS10m'] * (hub_height / 10) ** 0.14)
                            theo_power = wind_power_curve_realistic(hub_speed, wind_capacity)
                            theoretical_power.append(theo_power)
                            
                            # Instantaneous capacity factor
                            cf = row['Wind_Energy_kWh'] / wind_capacity if wind_capacity > 0 else 0
                            capacity_factor_inst.append(cf)
                        
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=theoretical_power, 
                                     name="Theoretical Power Output", 
                                     line=dict(color='purple', width=2, dash='dot')),
                            row=3, col=1, secondary_y=False
                        )
                        
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=actual_power, 
                                     name="Actual Power Output", 
                                     line=dict(color='magenta', width=2)),
                            row=3, col=1, secondary_y=False
                        )
                        
                        wind_fig.add_trace(
                            go.Scatter(x=wind_x_col, y=capacity_factor_inst, 
                                     name="Instantaneous Capacity Factor", 
                                     line=dict(color='brown', width=2)),
                            row=3, col=1, secondary_y=True
                        )
                        
                        # Update y-axis labels for wind
                        wind_fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=1, secondary_y=False)
                        wind_fig.update_yaxes(title_text="Energy Production (kWh)", row=1, col=1, secondary_y=True)
                        wind_fig.update_yaxes(title_text="Air Density (kg/m³)", row=2, col=1, secondary_y=False)
                        wind_fig.update_yaxes(title_text="Temperature (°C) / Direction Change (°)", row=2, col=1, secondary_y=True)
                        wind_fig.update_yaxes(title_text="Power Output (kW)", row=3, col=1, secondary_y=False)
                        wind_fig.update_yaxes(title_text="Capacity Factor", row=3, col=1, secondary_y=True, range=[0, 1])
                        
                        wind_fig.update_layout(
                            height=900, 
                            title_text=f"Wind System Performance Analysis - {wind_title_suffix}",
                            showlegend=True
                        )
                        
                        wind_fig.update_xaxes(tickangle=45)
                        
                        st.plotly_chart(wind_fig, use_container_width=True)
                        
                        # Wind performance statistics
                        st.subheader("📊 Wind Performance Statistics")
                        
                        wind_stats_col1, wind_stats_col2, wind_stats_col3, wind_stats_col4 = st.columns(4)
                        
                        with wind_stats_col1:
                            st.metric("Avg Wind Speed (10m)", f"{wind_plot_data['WS10m'].mean():.1f} m/s")
                            if 'Wind_Speed_Hub' in wind_plot_data.columns:
                                st.metric(f"Avg Wind Speed ({hub_height}m)", f"{wind_plot_data['Wind_Speed_Hub'].mean():.1f} m/s")
                        
                        with wind_stats_col2:
                            st.metric("Avg Power Output", f"{wind_plot_data['Wind_Energy_kWh'].mean():.1f} kW")
                            st.metric("Peak Power Output", f"{wind_plot_data['Wind_Energy_kWh'].max():.1f} kW")
                        
                        with wind_stats_col3:
                            avg_cf = np.mean(capacity_factor_inst)
                            st.metric("Avg Capacity Factor", f"{avg_cf:.1%}")
                            st.metric("Wind Variability", f"{wind_plot_data['WS10m'].std():.1f} m/s")
                        
                        with wind_stats_col4:
                            st.metric("Power Variance", f"{wind_plot_data['Wind_Energy_kWh'].std():.1f} kW")
                            st.metric("Avg Air Density", f"{np.mean(air_densities):.3f} kg/m³")
                    
                    # Combined analysis for both technologies
                    if (solar_results and 'hourly_data' in solar_results) and (wind_results and 'hourly_data' in wind_results):
                        st.subheader("⚡ Combined Solar + Wind Analysis")
                        
                        # Create combined energy production chart
                        combined_fig = go.Figure()
                        
                        # Use solar data timeframe for consistency
                        if time_period == "Daily (24 hours)":
                            combined_data_solar = solar_results['hourly_data'].head(24)
                            combined_data_wind = wind_results['hourly_data'].head(24)
                            combined_x = combined_data_solar.index.strftime('%H:%M')
                        elif time_period == "Weekly (7 days)":
                            combined_data_solar = solar_results['hourly_data'].head(168)
                            combined_data_wind = wind_results['hourly_data'].head(168)
                            combined_x = combined_data_solar.index.strftime('%m-%d %H')
                        elif time_period == "Monthly (30 days)":
                            combined_data_solar = solar_results['hourly_data'].head(720)
                            combined_data_wind = wind_results['hourly_data'].head(720)
                            combined_x = combined_data_solar.index.strftime('%m-%d')
                        else:
                            combined_data_solar = solar_results['hourly_data']
                            combined_data_wind = wind_results['hourly_data']
                            combined_x = combined_data_solar.index.strftime('%m-%d')
                        
                        # Align the datasets
                        min_length = min(len(combined_data_solar), len(combined_data_wind), len(combined_x))
                        
                        combined_fig.add_trace(go.Scatter(
                            x=combined_x[:min_length],
                            y=combined_data_wind['Wind_Energy_kWh'][:min_length],
                            mode='lines',
                            name='Wind Energy',
                            line=dict(color='lightblue', width=2),
                            stackgroup='one'
                        ))
                        
                        # Calculate total combined energy
                        total_energy = (combined_data_solar['Solar_Energy_kWh'][:min_length] + 
                                      combined_data_wind['Wind_Energy_kWh'][:min_length]).tolist()
                        
                        combined_fig.add_trace(go.Scatter(
                            x=combined_x[:min_length],
                            y=total_energy,
                            mode='lines',
                            name='Total Combined Energy',
                            line=dict(color='purple', width=3, dash='dash'),
                            yaxis='y2'
                        ))
                        
                        # Update layout for combined chart
                        combined_fig.update_layout(
                            title=f'Combined Solar + Wind Energy Production - {time_period}',
                            xaxis_title='Time',
                            yaxis=dict(title='Individual Technology Output (kWh)'),
                            yaxis2=dict(title='Total Combined Output (kWh)', overlaying='y', side='right'),
                            height=500,
                            hovermode='x unified'
                        )
                        
                        combined_fig.update_xaxes(tickangle=45)
                        st.plotly_chart(combined_fig, use_container_width=True)
                        
                        # Complementarity analysis
                        st.subheader("🔄 Energy Complementarity Analysis")
                        
                        # Calculate correlation between solar and wind
                        solar_energy = combined_data_solar['Solar_Energy_kWh'][:min_length]
                        wind_energy = combined_data_wind['Wind_Energy_kWh'][:min_length]
                        
                        correlation = np.corrcoef(solar_energy, wind_energy)[0, 1] if min_length > 1 else 0
                        
                        # Calculate complementarity metrics
                        solar_cf = solar_energy.sum() / (solar_results['capacity_kw'] * min_length) if min_length > 0 else 0
                        wind_cf = wind_energy.sum() / (wind_results['capacity_kw'] * min_length) if min_length > 0 else 0
                        combined_cf = (solar_energy.sum() + wind_energy.sum()) / ((solar_results['capacity_kw'] + wind_results['capacity_kw']) * min_length) if min_length > 0 else 0
                        
                        # Smoothing effect (coefficient of variation)
                        solar_cv = solar_energy.std() / solar_energy.mean() if solar_energy.mean() > 0 else 0
                        wind_cv = wind_energy.std() / wind_energy.mean() if wind_energy.mean() > 0 else 0
                        combined_cv = np.array(total_energy).std() / np.array(total_energy).mean() if np.array(total_energy).mean() > 0 else 0
                        
                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                        
                        with comp_col1:
                            st.metric("Solar-Wind Correlation", f"{correlation:.3f}")
                            correlation_desc = "Strong anti-correlation" if correlation < -0.5 else "Moderate anti-correlation" if correlation < -0.2 else "Weak correlation" if abs(correlation) < 0.2 else "Positive correlation"
                            st.caption(correlation_desc)
                        
                        with comp_col2:
                            st.metric("Combined Capacity Factor", f"{combined_cf:.1%}")
                            st.caption(f"Solar: {solar_cf:.1%}, Wind: {wind_cf:.1%}")
                        
                        with comp_col3:
                            smoothing_benefit = ((solar_cv + wind_cv) / 2 - combined_cv) / ((solar_cv + wind_cv) / 2) * 100 if (solar_cv + wind_cv) > 0 else 0
                            st.metric("Output Smoothing Benefit", f"{smoothing_benefit:.1f}%")
                            st.caption("Variability reduction vs individual")
                        
                        with comp_col4:
                            # Calculate peak shaving potential
                            max_individual = max(solar_energy.max(), wind_energy.max())
                            max_combined = max(total_energy)
                            peak_ratio = max_combined / max_individual if max_individual > 0 else 1
                            st.metric("Peak Demand Ratio", f"{peak_ratio:.2f}")
                            st.caption("Combined peak vs individual peak")
                        
                        # Energy mix visualization
                        st.subheader("📊 Energy Mix Analysis")
                        
                        # Create pie chart for energy contribution
                        total_solar_energy = solar_energy.sum()
                        total_wind_energy = wind_energy.sum()
                        
                        pie_fig = go.Figure(data=[go.Pie(
                            labels=['Solar Energy', 'Wind Energy'],
                            values=[total_solar_energy, total_wind_energy],
                            hole=0.3,
                            marker_colors=['orange', 'lightblue']
                        )])
                        
                        pie_fig.update_layout(
                            title=f"Energy Contribution Mix - {time_period}",
                            height=400
                        )
                        
                        st.plotly_chart(pie_fig, use_container_width=True)
                        
                        # Time-based analysis
                        if time_period in ["Daily (24 hours)", "Weekly (7 days)"]:
                            st.subheader("⏰ Hourly Production Patterns")
                            
                            # Group by hour of day for pattern analysis
                            hourly_solar = combined_data_solar.groupby(combined_data_solar.index.hour)['Solar_Energy_kWh'].mean()
                            hourly_wind = combined_data_wind.groupby(combined_data_wind.index.hour)['Wind_Energy_kWh'].mean()
                            
                            pattern_fig = go.Figure()
                            
                            pattern_fig.add_trace(go.Scatter(
                                x=list(range(24)),
                                y=hourly_solar.values,
                                mode='lines+markers',
                                name='Average Solar Production',
                                line=dict(color='orange', width=3),
                                marker=dict(size=6)
                            ))
                            
                            pattern_fig.add_trace(go.Scatter(
                                x=list(range(24)),
                                y=hourly_wind.values,
                                mode='lines+markers',
                                name='Average Wind Production',
                                line=dict(color='lightblue', width=3),
                                marker=dict(size=6)
                            ))
                            
                            pattern_fig.update_layout(
                                title='Average Hourly Production Patterns',
                                xaxis_title='Hour of Day',
                                yaxis_title='Average Energy Production (kWh)',
                                height=400,
                                xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                            )
                            
                            st.plotly_chart(pattern_fig, use_container_width=True)
                            
                            # Peak production times analysis
                            solar_peak_hour = hourly_solar.idxmax()
                            wind_peak_hour = hourly_wind.idxmax()
                            
                            st.info(f"**Production Peak Times**: Solar peaks at {solar_peak_hour:02d}:00, Wind peaks at {wind_peak_hour:02d}:00. "
                                   f"Time offset: {abs(solar_peak_hour - wind_peak_hour)} hours.")
                    
                    # Resource assessment and recommendations
                    st.subheader("📋 Time Series Analysis Summary & Recommendations")
                    
                    recommendations = []
                    
                    if solar_results and 'hourly_data' in solar_results:
                        avg_pr = np.mean(inst_pr) if 'inst_pr' in locals() else 0
                        if avg_pr < 0.7:
                            recommendations.append("⚠️ **Solar Performance**: Low performance ratio detected. Consider system maintenance, cleaning, or shading analysis.")
                        elif avg_pr > 0.85:
                            recommendations.append("✅ **Solar Performance**: Excellent performance ratio. System operating optimally.")
                        
                        solar_data_quality = plot_data if 'plot_data' in locals() else solar_results['hourly_data'].head(24)
                        temp_stress_hours = len(solar_data_quality[solar_data_quality['T2m'] > 40])
                        if temp_stress_hours > len(solar_data_quality) * 0.3:
                            recommendations.append("🌡️ **Temperature Impact**: High temperature exposure detected. Consider enhanced cooling or tracking systems.")
                    
                    if wind_results and 'hourly_data' in wind_results:
                        avg_wind_cf = wind_results.get('capacity_factor_actual', 0)
                        if avg_wind_cf < 0.25:
                            recommendations.append("⚠️ **Wind Performance**: Low capacity factor. Consider higher hub height or different turbine model.")
                        elif avg_wind_cf > 0.4:
                            recommendations.append("✅ **Wind Performance**: Excellent wind resource utilization.")
                        
                        wind_data_quality = wind_plot_data if 'wind_plot_data' in locals() else wind_results['hourly_data'].head(24)
                        low_wind_hours = len(wind_data_quality[wind_data_quality['WS10m'] < 3])
                        if low_wind_hours > len(wind_data_quality) * 0.4:
                            recommendations.append("💨 **Wind Resource**: Significant low wind periods detected. Consider energy storage or grid backup.")
                    
                    if (solar_results and 'hourly_data' in solar_results) and (wind_results and 'hourly_data' in wind_results):
                        if correlation < -0.3:
                            recommendations.append("🔄 **Complementarity**: Strong complementary behavior detected. Excellent for grid stability and reduced storage needs.")
                        elif smoothing_benefit > 15:
                            recommendations.append("📈 **Output Smoothing**: Significant variability reduction achieved through technology combination.")
                    
                    # Display recommendations
                    if recommendations:
                        for rec in recommendations:
                            st.write(rec)
                    else:
                        st.info("📊 System performance analysis shows normal operating conditions within expected parameters.")
                    
                    # Data export options
                    # Data export options
                    st.subheader("📥 Export Time Series Data")
                    export_col1, export_col2, export_col3 = st.columns(3)

                    with export_col1:
                        if solar_results and 'hourly_data' in solar_results:
                            csv_data_solar = solar_results['hourly_data'].to_csv(index=True)
                            st.download_button(
                                label="📊 Download Solar Data CSV",
                                data=csv_data_solar,
                                file_name=f"solar_timeseries_{selected_location['lat']:.2f}_{selected_location['lon']:.2f}.csv",
                                mime='text/csv',
                                key="download_solar_csv"
                            )
                        else:
                            st.button("📊 Download Solar Data CSV", disabled=True)

                    with export_col2:
                        if wind_results and 'hourly_data' in wind_results:
                            csv_data_wind = wind_results['hourly_data'].to_csv(index=True)
                            st.download_button(
                                label="💨 Download Wind Data CSV",
                                data=csv_data_wind,
                                file_name=f"wind_timeseries_{selected_location['lat']:.2f}_{selected_location['lon']:.2f}.csv",
                                mime='text/csv',
                                key="download_wind_csv"
                            )
                        else:
                            st.button("💨 Download Wind Data CSV", disabled=True)

                    with export_col3:
                        if (solar_results and 'hourly_data' in solar_results) and (wind_results and 'hourly_data' in wind_results):
                            # Get both dataframes
                            solar_df = solar_results['hourly_data'].copy()
                            wind_df = wind_results['hourly_data'].copy()
                            
                            # Ensure both dataframes have the same index (datetime)
                            # Merge on index to combine the data
                            combined_df = pd.merge(solar_df, wind_df, left_index=True, right_index=True, how='outer', suffixes=('_solar', '_wind'))
                            
                            # Convert to CSV
                            csv_data_combined = combined_df.to_csv(index=True)
                            
                            st.download_button(
                                label="⚡ Download Combined Data CSV",
                                data=csv_data_combined,
                                file_name=f"combined_timeseries_{selected_location['lat']:.2f}_{selected_location['lon']:.2f}.csv",
                                mime='text/csv',
                                key="download_combined_csv"
                            )
                        else:
                            st.button("⚡ Download Combined Data CSV", disabled=True)
        
        else:
            st.error("Please select at least one energy type and ensure valid system specifications.")
    
    else:
        # Enhanced welcome section
        st.markdown("""
        ## 🚀 Enhanced Renewable Energy Assessment Tool
        
        This advanced tool provides physics-based analysis of renewable energy systems in the Arabian Peninsula using:
        
        ### 🔬 Enhanced Calculation Methods:
        - **Solar Analysis**: Real sun-angle calculations, panel orientation optimization, temperature effects
        - **Wind Analysis**: Hub height corrections, air density effects, realistic power curves
        - **Economic Modeling**: NPV, LCOE, lifecycle costs including O&M and replacements
        - **Environmental Assessment**: Lifecycle CO₂ analysis including manufacturing emissions
        
        ### 🗺️ Interactive Map Interface:
        - Click anywhere on the Arabian Peninsula to select your location
        - Real-time coordinate display and weather data integration
        - TMY (Typical Meteorological Year) data support for accurate modeling
        
        ### 📊 Advanced Features:
        - Performance ratio calculations for solar systems
        - Wind shear and air density corrections
        - Economic analysis with NPV and payback calculations
        - Comprehensive technical performance metrics
        - Time series analysis with detailed weather correlations
        
        ### 🎯 Key Improvements Over Basic Tools:
        1. **Physics-Based Solar Modeling**: Accounts for sun angles, panel tilt, azimuth optimization
        2. **Realistic Wind Modeling**: Hub height corrections, turbulence, air density effects  
        3. **Enhanced Economics**: Includes O&M, degradation, inverter replacements, NPV analysis
        4. **Environmental Accuracy**: Lifecycle emissions including manufacturing footprint
        5. **Interactive Location Selection**: Map-based coordinate picking for any location
        
        **Select your location on the map and configure your system parameters to get started!** 👈
        """)

if __name__ == "__main__":
    main()

