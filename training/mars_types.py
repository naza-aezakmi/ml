"""Train ten terrain prototypes and ship a compact Mars click lookup with fixed Earth catalogs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import joblib
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/mars_types';OUT=ROOT/'reports/global_terrain';PUBLIC=ROOT/'public'
FIELDS=['median_slope_deg','p90_slope_deg','relief_p95_p05_m']

def xyz(lat,lon,radius=1):
    a,b=np.radians(lat),np.radians(lon)
    return radius*np.column_stack([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)])

def descriptors(body):
    cache=DATA/f'{body}_descriptors.npz'
    if cache.exists():return dict(np.load(cache))
    source=dict(np.load(ROOT/f'data/global_terrain/{body}_cells.npz'))
    radius=3396000 if body=='mars' else 6371008.8
    points=xyz(source['lat'],source['lon'],radius)
    tree=cKDTree(points)
    vectors=[];valid=[]
    for start in range(0,len(points),20000):
        distance,ids=tree.query(points[start:start+20000],k=81,workers=2)
        s=source['slope_deg'][ids];z=source['elevation_m'][ids]
        vectors.append(np.column_stack([np.median(s,axis=1),np.percentile(s,90,axis=1),
                                       np.percentile(z,95,axis=1)-np.percentile(z,5,axis=1)]).astype('float32'))
        valid.append(distance[:,-1]<65000)
        if start%200000==0:print(body,start,'/',len(points),flush=True)
    result={k:source[k] for k in ['lat','lon','area_km2']}
    result['features']=np.concatenate(vectors);result['valid']=np.concatenate(valid)
    np.savez_compressed(cache,**result)
    return result

def name_type(slope,relief):
    shape=('Почти равнина' if slope<.2 else 'Пологий рельеф' if slope<.7 else
           'Увалистый рельеф' if slope<1.5 else 'Холмистый рельеф' if slope<3 else 'Сильно пересечённый рельеф')
    amplitude=('малый перепад' if relief<100 else 'умеренный перепад' if relief<400 else
               'выраженный перепад' if relief<1200 else 'большой перепад')
    return shape+' — '+amplitude

def main():
    DATA.mkdir(exist_ok=True,parents=True);PUBLIC.mkdir(exist_ok=True)
    mars=descriptors('mars');earth=descriptors('earth')
    rng=np.random.default_rng(42)
    good=np.flatnonzero(mars['valid'])
    weights=mars['area_km2'][good].astype(float);weights/=weights.sum()
    train=rng.choice(good,min(150000,len(good)),replace=False,p=weights)
    scaler=RobustScaler().fit(np.log1p(mars['features'][train]))
    x=scaler.transform(np.log1p(mars['features'][train]))
    validation=rng.choice(len(x),2000,replace=False)
    diagnostics=[];models={}
    with threadpool_limits(limits=2):
        for k in [6,10,14]:
            model=MiniBatchKMeans(n_clusters=k,n_init=5,batch_size=4096,random_state=42).fit(x)
            score=silhouette_score(x[validation],model.predict(x[validation]))
            diagnostics.append({'k':k,'silhouette':float(score),'inertia':float(model.inertia_)})
            models[k]=model
            print('clusters',k,'silhouette',score,flush=True)
    model=models[10]
    order=np.argsort(model.cluster_centers_[:,0]+model.cluster_centers_[:,2])
    remap=np.empty(10,dtype='uint8');remap[order]=np.arange(10)
    mx=scaler.transform(np.log1p(mars['features']))
    with threadpool_limits(limits=2): labels=remap[model.predict(mx)]
    earth_ids=np.flatnonzero(earth['valid'])
    ex=scaler.transform(np.log1p(earth['features'][earth_ids]))
    types=[]
    for type_id,old_id in enumerate(order):
        mask=(labels==type_id)&mars['valid']
        center=model.cluster_centers_[old_id]
        distance=np.linalg.norm(ex-center,axis=1)
        picks=[]
        for index in np.argsort(distance)[:20000]:
            idx=earth_ids[index]
            loc=xyz(np.array([earth['lat'][idx]]),np.array([earth['lon'][idx]]))[0]
            if all(np.arccos(np.clip(np.dot(loc,p['_xyz']),-1,1))*6371>500 for p in picks):
                picks.append({'id':f'T{type_id+1:02d}_E{len(picks)+1}',
                              'lat':float(earth['lat'][idx]),'lon':float(earth['lon'][idx]),
                              'distance':float(distance[index]),'features':dict(zip(FIELDS,map(float,earth['features'][idx]))),
                              'landcover_status':'not_screened_for_this_coordinate','_xyz':loc})
            if len(picks)==3:break
        for p in picks:p.pop('_xyz')
        values=mars['features'][mask]
        med=np.median(values,axis=0)
        radius95=float(np.quantile(np.linalg.norm(mx[mask]-center,axis=1),.95))
        for p in picks:p['outside_mars_type_radius95']=p['distance']>radius95
        representative=np.flatnonzero(mask)[np.argmin(np.linalg.norm(mx[mask]-center,axis=1))]
        types.append({'id':int(type_id),'label':f'{type_id+1:02d}. '+name_type(med[0],med[2]),
                      'features_median':dict(zip(FIELDS,map(float,med))),
                      'features_p10':dict(zip(FIELDS,map(float,np.percentile(values,10,axis=0)))),
                      'features_p90':dict(zip(FIELDS,map(float,np.percentile(values,90,axis=0)))),
                      'area_fraction':float(mars['area_km2'][mask].sum()/mars['area_km2'][mars['valid']].sum()),
                      'representative':{'lat':float(mars['lat'][representative]),'lon':float(mars['lon'][representative])},
                      'earth_candidates':picks,'radius95':radius95,
                      'temperature_c':None,'humidity':None,'precipitation':None,
                      'surface_material':'not_inferred_from_DEM'})
    # Nearest spherical cell, not flat latitude/longitude distance; no missing polar patch lookup.
    lat=90-(np.arange(360)+.5)*.5;lon=-180+(np.arange(720)+.5)*.5
    lo,la=np.meshgrid(lon,lat)
    nearest=cKDTree(xyz(mars['lat'][good],mars['lon'][good])).query(xyz(la.ravel(),lo.ravel()),workers=2)[1]
    grid=labels[good[nearest]].astype('uint8')
    (PUBLIC/'mars_types_grid.bin').write_bytes(grid.tobytes())
    np.savez_compressed(DATA/'mars_type_labels.npz',labels=labels)
    metadata={'version':1,'body':'Mars','model':'MiniBatchKMeans','n_types':10,'seed':42,
        'decision':'10 prototypes chosen for demo UX; not a claim of 10 natural geological classes',
        'fields':FIELDS,'features_note':'81 nearest 10 km cells; typically ~50 km radius; require furthest neighbor <65 km. No absolute elevation cross-planet matching.',
        'mars_valid_area_fraction':float(mars['area_km2'][mars['valid']].sum()/mars['area_km2'].sum()),
        'grid':{'width':720,'height':360,'step_degrees':.5,'north':90,'west':-180,'dtype':'uint8','row_order':'north_to_south','url':'mars_types_grid.bin'},
        'diagnostics':diagnostics,'types':types,'sources':['NASA MOLA MEGDR32','NOAA ETOPO2022 60s'],
        'limits':['No soil, rock abundance, humidity or rainfall inferred from terrain',
                  'Earth matches explain regional morphology only; landcover/logistics not screened for new coordinates',
                  'All clicks in a type return same Earth catalog; per-click optimality is not claimed',
                  '0.5-degree lookup approximates original 10 km grid; boundaries are approximate'],
        'climate_status':'Additional layers are separate from terrain clustering and analog matching.'}
    (PUBLIC/'mars_types.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    with gzip.open(PUBLIC/'mars_types_grid.bin.gz','wb') as f:f.write(grid.tobytes())
    joblib.dump({'scaler':scaler,'model':model,'remap':remap},ROOT/'models/mars_terrain_types.joblib')
    pd.DataFrame([{'id':t['id'], 'label':t['label'],'area_fraction':t['area_fraction'],**t['features_median']} for t in types]).to_csv(OUT/'mars_types.csv',index=False)
    print('Ready:',PUBLIC/'mars_types.json',flush=True)

if __name__=='__main__':main()
