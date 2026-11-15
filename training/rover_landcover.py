"""Add ESA WorldCover context to terrain candidates; no conversion into wheel slip labels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window

from global_terrain_data import ROOT, sha256
from terrain_candidates import crs_for

DATA=ROOT/'data/rover/worldcover'
OUT=ROOT/'reports/global_terrain'
CLASSES={10:'trees',20:'shrubland',30:'grassland',40:'cropland',50:'built_up',
         60:'bare_sparse',70:'snow_ice',80:'permanent_water',90:'herbaceous_wetland',
         95:'mangroves',100:'moss_lichen'}
GRID=from_origin(-50000,50000,100,100)


def tile_url(lat,lon):
    tag=f"{'N' if lat>=0 else 'S'}{abs(lat):02d}{'E' if lon>=0 else 'W'}{abs(lon):03d}"
    return tag,f'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tag}_Map.tif'


def download_site(site):
    ident=site['id']
    destination=DATA/f'{ident}.tif'
    meta=destination.with_suffix('.json')
    if destination.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if sha256(destination)!=record['sha256']: raise ValueError('WorldCover checksum mismatch')
        return record
    crs=crs_for(site)
    bounds=transform_bounds(crs,'EPSG:4326',-50000,-50000,50000,50000,densify_pts=41)
    left,bottom,right,top=bounds
    target=np.zeros((1000,1000),dtype='uint8')
    sources=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                      GDAL_HTTP_TIMEOUT='45',GDAL_HTTP_MAX_RETRY='2',GDAL_HTTP_RETRY_DELAY='2'):
        for lat in range(3*math.floor(bottom/3),3*math.ceil(top/3),3):
            for lon in range(3*math.floor(left/3),3*math.ceil(right/3),3):
                tag,url=tile_url(lat,lon)
                try:
                    dataset=rasterio.open(url)
                except rasterio.errors.RasterioIOError as error:
                    if '404' in str(error):
                        sources.append({'url':url,'tile':tag,'status':'missing_404'})
                        continue
                    raise
                with dataset as src:
                    b=(max(left,src.bounds.left),max(bottom,src.bounds.bottom),
                       min(right,src.bounds.right),min(top,src.bounds.top))
                    raw=from_bounds(*b,transform=src.transform)
                    col,row=max(0,math.floor(raw.col_off)-2),max(0,math.floor(raw.row_off)-2)
                    width=min(src.width-col,math.ceil(raw.col_off+raw.width)+2-col)
                    height=min(src.height-row,math.ceil(raw.row_off+raw.height)+2-row)
                    if min(width,height)<=0: continue
                    window=Window(col,row,width,height)
                    shape=(max(1,math.ceil(height/10)),max(1,math.ceil(width/10)))
                    # Categorical nearest resampling; this is screening at ~100 m, not a 10 m pixel census.
                    z=src.read(1,window=window,out_shape=shape,resampling=Resampling.nearest)
                    unknown=set(np.unique(z))-set(CLASSES)-{0}
                    if unknown: raise ValueError(f'Unexpected WorldCover class codes: {unknown}')
                    transform=src.window_transform(window)*Affine.scale(width/shape[1],height/shape[0])
                    reproject(z,target,src_transform=transform,src_crs=src.crs,src_nodata=0,
                              dst_transform=GRID,dst_crs=crs,dst_nodata=0,
                              resampling=Resampling.nearest,init_dest_nodata=False)
                    sources.append({'url':url,'tile':tag,'native_pixel_size_degrees':list(src.res),
                                    'native_window':[col,row,width,height], 'read_shape':list(shape),
                                    'available_overviews':src.overviews(1)})
    if (target!=0).mean()<.95: raise ValueError(f'{ident}: insufficient WorldCover coverage')
    DATA.mkdir(parents=True,exist_ok=True)
    with rasterio.open(destination,'w',driver='GTiff',height=1000,width=1000,count=1,dtype='uint8',
                       transform=GRID,crs=crs,nodata=0,compress='deflate') as dst:
        dst.write(target,1)
    record={'site_id':ident,'sources':sources,'sha256':sha256(destination),
            'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'product':'ESA WorldCover 2021 v200',
            'original_nominal_resolution_m':10,'analysis_pixel_m':100,'window_km':100,
            'resampling':'nearest categorical screening; source overviews may be used',
            'valid_fraction':float((target!=0).mean())}
    meta.write_text(json.dumps(record,indent=2))
    print(f'WorldCover {ident}: {len(sources)} tiles',flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['download','summarize'])
    args=parser.parse_args()
    plan=json.loads((ROOT/'data/candidates/plan.json').read_text())
    sites=plan['candidates']
    DATA.mkdir(parents=True,exist_ok=True)
    if args.stage=='download':
        with ThreadPoolExecutor(max_workers=3) as pool:
            records=list(pool.map(download_site,sites))
        (DATA/'manifest.json').write_text(json.dumps(records,indent=2))
    else:
        rows=[]
        for site in sites:
            with rasterio.open(DATA/f"{site['id']}.tif") as src: z=src.read(1)
            valid=z!=0
            row={'site_id':site['id'],'target_id':site['target_id'],'lat':site['lat'],'lon':site['lon'],
                 'valid_fraction':float(valid.mean())}
            for code,name in CLASSES.items(): row[name+'_fraction']=float((z[valid]==code).mean())
            row['vegetated_fraction']=sum(row[CLASSES[c]+'_fraction'] for c in [10,20,30,90,95,100])
            # Descriptive land-cover similarity screen, not a predicted probability of rover success.
            row['mars_surface_screen']='bare_sparse_dominant' if row['bare_sparse_fraction']>=.5 else 'mixed_or_other_cover'
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'rover_landcover.csv',index=False)
        print(pd.DataFrame(rows)[['site_id','bare_sparse_fraction','cropland_fraction','vegetated_fraction','snow_ice_fraction']].round(3).to_string(index=False))


if __name__=='__main__': main()
