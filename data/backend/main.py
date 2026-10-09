from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3, json, math
from pathlib import Path

DB = Path(__file__).parent / 'analog.db'
app = FastAPI(title='AstroAnalog AI API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])
SEEDS = [
('Atacama Desert','Chile',-23.5,-68.2,'mars',0.97,0.86,0.92,0.35,'Hyper-arid terrain and exposed mineral surfaces; useful for rover and biosignature studies.'),
('McMurdo Dry Valleys','Antarctica',-77.5,162.0,'both',0.90,0.80,0.76,0.96,'Cold, dry polar desert; useful for extreme-environment field operations.'),
('Lanzarote','Spain',29.05,-13.62,'both',0.65,0.94,0.87,0.40,'Basaltic lava fields useful for rover mobility and geological field training.'),
('Haughton Crater','Canada',75.38,-89.67,'both',0.77,0.91,0.82,0.88,'Impact crater terrain with polar conditions and exploration analog history.'),
('Death Valley','United States',36.46,-116.87,'mars',0.91,0.78,0.89,0.20,'Arid salt flats and rugged landscapes useful for mobility testing.'),
('Mauna Kea','United States',19.82,-155.47,'moon',0.70,0.97,0.93,0.46,'Volcanic rocky terrain useful for simulated lunar surface operations.'),
('Wadi Rum','Jordan',29.58,35.42,'mars',0.89,0.83,0.88,0.29,'Desert sandstone landscapes for exploration and operational simulations.'),
('Iceland Highlands','Iceland',64.7,-18.5,'moon',0.62,0.96,0.90,0.77,'Basaltic volcanic terrain useful for planetary field geology training.')]
TARGETS = {'mars': {'aridity':0.97,'geology':0.86,'terrain':0.88,'cold':0.80}, 'moon':{'aridity':0.92,'geology':0.97,'terrain':0.92,'cold':0.90}}
WEIGHTS = {'aridity':0.24,'geology':0.31,'terrain':0.29,'cold':0.16}

def connect():
    conn=sqlite3.connect(DB)
    conn.row_factory=sqlite3.Row
    return conn

def initialize():
    with connect() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS sites (id INTEGER PRIMARY KEY, name TEXT UNIQUE, country TEXT, lat REAL, lon REAL, category TEXT, aridity REAL, geology REAL, terrain REAL, cold REAL, description TEXT)')
        conn.executemany('INSERT OR IGNORE INTO sites (name,country,lat,lon,category,aridity,geology,terrain,cold,description) VALUES (?,?,?,?,?,?,?,?,?,?)',SEEDS)
initialize()

class AnalyzeRequest(BaseModel):
    target: str = 'mars'
    weights: dict[str,float] | None = None

@app.get('/api/health')
def health(): return {'status':'ok','mode':'demonstration','database':'sqlite'}

@app.get('/api/sites')
def sites():
    with connect() as conn: return [dict(r) for r in conn.execute('SELECT * FROM sites ORDER BY name')]

@app.get('/api/targets')
def targets(): return TARGETS

@app.post('/api/analyze')
def analyze(payload: AnalyzeRequest):
    if payload.target not in TARGETS: raise HTTPException(400,'Target must be moon or mars')
    weights=payload.weights or WEIGHTS
    if set(weights)!=set(WEIGHTS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
        raise HTTPException(400,'Weights must be nonnegative finite numbers for aridity, geology, terrain, cold with a positive total')
    total=sum(weights.values()); weights={k:v/total for k,v in weights.items()}
    results=[]
    for site in sites():
        contributions={k:round((1-abs(site[k]-TARGETS[payload.target][k]))*weights[k]*100,2) for k in WEIGHTS}
        site['score']=round(sum(contributions.values()),1)
        site['contributions']=contributions
        site['differences']={k:round(abs(site[k]-TARGETS[payload.target][k]),2) for k in WEIGHTS}
        results.append(site)
    results.sort(key=lambda x:x['score'],reverse=True)
    return {'target':payload.target,'weights':weights,'results':results,'disclaimer':'Illustrative normalized feature values and heuristic scoring. Not validated NASA-derived measurements or suitability predictions.'}


import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)