from django.shortcuts import render
import requests
from django.conf import settings

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# from django.http import HttpResponse

# def home(request):
#     return render(request,'website\\index.html')
#     # return HttpResponse('works')          for testing

def about(request):
    return render(request,'website//about.html')

# weather data fetch
def get_weather_data():
    city = "Kathmandu"
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def home(request):
    weather_data = get_weather_data()
    aqi_data = get_aqi_data()
    
    context = {
        'weather': weather_data,
        'aqi': aqi_data,
        'weather_api_key': settings.WEATHER_API_KEY,
    }
    return render(request, 'website/index.html', context)

# aqi data fetch

def calculate_aqi_pm25(pm25):
    """Calculate AQI for PM2.5"""
    if pm25 <= 12.0:
        return linear_interpolation(0, 50, 0, 12.0, pm25)
    elif pm25 <= 35.4:
        return linear_interpolation(51, 100, 12.1, 35.4, pm25)
    elif pm25 <= 55.4:
        return linear_interpolation(101, 150, 35.5, 55.4, pm25)
    elif pm25 <= 150.4:
        return linear_interpolation(151, 200, 55.5, 150.4, pm25)
    elif pm25 <= 250.4:
        return linear_interpolation(201, 300, 150.5, 250.4, pm25)
    elif pm25 <= 350.4:
        return linear_interpolation(301, 400, 250.5, 350.4, pm25)
    else:
        return linear_interpolation(401, 500, 350.5, 500.4, pm25)

def calculate_aqi_o3(o3):
    """Calculate AQI for O3 (in µg/m³)"""
    # Convert µg/m³ to ppb (parts per billion)
    o3_ppb = o3 / 1.96
    
    if o3_ppb <= 54:
        return linear_interpolation(0, 50, 0, 54, o3_ppb)
    elif o3_ppb <= 70:
        return linear_interpolation(51, 100, 55, 70, o3_ppb)
    elif o3_ppb <= 85:
        return linear_interpolation(101, 150, 71, 85, o3_ppb)
    elif o3_ppb <= 105:
        return linear_interpolation(151, 200, 86, 105, o3_ppb)
    elif o3_ppb <= 200:
        return linear_interpolation(201, 300, 106, 200, o3_ppb)
    else:
        return linear_interpolation(301, 500, 201, 600, o3_ppb)

def linear_interpolation(aqi_low, aqi_high, conc_low, conc_high, conc):
    """Linear interpolation to calculate AQI"""
    return round(((aqi_high - aqi_low) / (conc_high - conc_low)) * (conc - conc_low) + aqi_low)

def get_aqi_category(aqi):
    """Get AQI category and color based on value"""
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

def get_aqi_data():
    lat = 27.7167
    lon = 85.3167
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Get PM2.5 and O3 values
        pm25 = data['list'][0]['components']['pm2_5']
        o3 = data['list'][0]['components']['o3']
        
        # Calculate AQI for both pollutants
        aqi_pm25 = calculate_aqi_pm25(pm25)
        aqi_o3 = calculate_aqi_o3(o3)
        
        # Overall AQI is the maximum of the two
        final_aqi = max(aqi_pm25, aqi_o3)
        
        # Get category and color
        category, color = get_aqi_category(final_aqi)
        
        # Add calculated values to the data
        data['list'][0]['main']['calculated_aqi'] = final_aqi
        data['list'][0]['main']['aqi_category'] = category
        data['list'][0]['main']['aqi_color'] = color
        
        return data
    else:
        return None
    
@login_required
def home(request):
    weather_data = get_weather_data()
    aqi_data = get_aqi_data()
    
    context = {
        'weather': weather_data,
        'aqi': aqi_data,
        'weather_api_key': settings.WEATHER_API_KEY,
    }
    return render(request, 'website/index.html', context)



