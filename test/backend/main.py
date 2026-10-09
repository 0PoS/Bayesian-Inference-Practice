from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rasterio
from rasterio.enums import Resampling
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

app = FastAPI(title='DEM Engine')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], # Allowing all for hackathon simplicity
    allow_methods=['*'],
    allow_headers=['*']
)

# 1. Define the data format the frontend will send
class DEMCompareRequest(BaseModel):
    earth_dem_path: str
    target_dem_path: str

# 2. Core math functions to extract data from the files
def extract_normalized_elevation(filepath: str, target_shape=(256, 256)):
    """Loads a GeoTIFF, standardizes its resolution, and normalizes heights to a 0-1 scale."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing DEM file: {filepath}")
        
    with rasterio.open(filepath) as src:
        # Resize to standard grid so Earth and Planet maps align perfectly
        data = src.read(1, out_shape=target_shape, resampling=Resampling.bilinear)
        
        # Handle missing data (NoData flags)
        data = np.nan_to_num(data, nan=np.nanmean(data))
        
        # Min-Max Normalization (maps all elevations between 0.0 and 1.0)
        d_min, d_max = np.min(data), np.max(data)
        if d_max == d_min: 
            return np.zeros(target_shape)
            
        return (data - d_min) / (d_max - d_min)

def compute_roughness(dem_array):
    """Calculates terrain steepness and ruggedness via gradient changes."""
    dy, dx = np.gradient(dem_array)
    return np.sqrt(dx**2 + dy**2)

# 3. The API Endpoint that ties it together
@app.post('/api/compare-terrain')
def compare_terrain(payload: DEMCompareRequest):
    try:
        # Load and normalize both maps
        earth_map = extract_normalized_elevation(payload.earth_dem_path)
        planet_map = extract_normalized_elevation(payload.target_dem_path)
        
        # Metric A: Structural Similarity (How visually similar are the craters/valleys?)
        # ssim returns a value between -1 and 1. We map it to a percentage.
        structure_score = ssim(earth_map, planet_map, data_range=1.0)
        structure_percent = max(0.0, structure_score * 100)
        
        # Metric B: Terrain Roughness (Does the Earth site have the same steepness/hazards?)
        earth_roughness = compute_roughness(earth_map)
        planet_roughness = compute_roughness(planet_map)
        
        # Calculate Mean Absolute Error (MAE) between the two roughness arrays
        roughness_error = float(np.mean(np.abs(earth_roughness - planet_roughness)))
        roughness_percent = max(0.0, (1.0 - roughness_error) * 100)
        
        # Composite Score (Weighting structural shape slightly higher than general roughness)
        final_score = (0.6 * structure_percent) + (0.4 * roughness_percent)
        
        # Return the dynamic output to the frontend
        return {
            "status": "success",
            "earth_file": payload.earth_dem_path,
            "target_file": payload.target_dem_path,
            "metrics": {
                "structural_similarity_pct": round(structure_percent, 2),
                "roughness_match_pct": round(roughness_percent, 2),
                "overall_terrain_score": round(final_score, 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)