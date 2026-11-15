"""Attach contextual Earth climatology; climate is deliberately outside the terrain match metric."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from global_terrain_data import ROOT

DATA=ROOT/'data/mars_types'

def point(site):
    path=DATA/f"earth_{site['id']}.json"
    if path.exists():payload=json.loads(path.read_text())
    else:
        url='https://power.larc.nasa.gov/api/temporal/climatology/point'
        params={'parameters':'T2M,PRECTOTCORR,RH2M,WS10M','community':'RE','latitude':site['lat'],'longitude':site['lon'],'format':'JSON'}
        r=requests.get(url,params=params,timeout=60);r.raise_for_status();payload=r.json()
        path.write_text(json.dumps(payload,indent=2))
    p=payload['properties']['parameter']
    vals={out:p[key]['ANN'] for out,key in [('air_temperature_2m_c','T2M'),('precipitation_mm_day','PRECTOTCORR'),('relative_humidity_pct','RH2M'),('wind_10m_m_s','WS10M')]}
    if any(v is None or v==-999 for v in vals.values()):raise ValueError('Missing POWER climatology')
    return site['id'],{**vals,'source':'NASA POWER climatology','temporal_note':'Earth long-term climatology; not directly matched to Mars noon seasonal model snapshot'}

def main():
    path=ROOT/'public/mars_types.json';meta=json.loads(path.read_text())
    sites=[s for t in meta['types'] for s in t['earth_candidates']]
    with ThreadPoolExecutor(max_workers=3) as pool: records=dict(pool.map(point,sites))
    for t in meta['types']:
        for s in t['earth_candidates']:s['climate']=records[s['id']]
        t['label']+=f" (p90≈{t['features_median']['p90_slope_deg']:.1f}°)" if 'p90≈' not in t['label'] else ''
    path.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('Added climate for',len(records),'Earth sites')

if __name__=='__main__':main()
