"""Parse MCD ASCII blocks, attach known context and package the static frontend data."""
import json,zipfile
import numpy as np
import rasterio
from global_terrain_data import ROOT,sha256
from rover_landcover import DATA as COVER,CLASSES

def main():
    public=ROOT/'public';path=public/'mars_types.json';m=json.loads(path.read_text())
    blocks=[];block=None
    for line in (ROOT/'data/mars_types/climate_0.txt').read_text().splitlines():
        if line.startswith('---- ||'):
            block={'latitudes':list(map(float,line.split('||')[1].split())),'longitudes':[],'rows':[]};blocks.append(block)
        elif '||' in line and block is not None:
            a,b=line.split('||');block['longitudes'].append(float(a));block['rows'].append(list(map(float,b.split())))
    if len(blocks)!=4:raise ValueError('Expected four documented MCD variable blocks')
    fields={}
    for block,key in zip(blocks,['surface_temperature_c','air_temperature_10m_c','pressure_10m_pa','wind_10m_m_s']):
        a=np.array(block['rows']).T
        assert a.shape==(len(block['latitudes']),len(block['longitudes'])) and np.isfinite(a).all()
        if key.endswith('_c'):a-=273.15
        if key.endswith('_pa'):assert (a>0).all()
        fields[key]=np.round(a,1).tolist()
    m['mars_climate']={'latitudes':blocks[0]['latitudes'],'longitudes':blocks[0]['longitudes'],
        'fields':fields,'source':'Mars Climate Database 6.2 model','context':{'Ls_degrees':0,'local_solar_time_hours':12,'atmospheric_height_m':10,'scenario':'climatology average solar'},
        'notes':'Nearest coarse model node; not present weather, annual mean, or terrain category. Humidity/precipitation not obtained.',
        'source_request':json.loads((ROOT/'data/mars_types/climate_source.json').read_text())}
    for t in m['types']:
        for e in t['earth_candidates']:
            p=COVER/f"{e['id']}.tif"
            if p.exists():
                with rasterio.open(p) as src:z=src.read(1)
                v=z[z!=0]
                e['landcover']={name+'_fraction':float((v==c).mean()) for c,name in CLASSES.items()}
                e['landcover']['vegetated_fraction']=sum(e['landcover'][CLASSES[c]+'_fraction'] for c in [10,20,30,90,95,100])
                e['landcover_status']='ESA WorldCover 2021, approximate 100m screening'
        # Same morphology score; order more field-like cover first, without claiming climate/soil equivalence.
        t['earth_candidates'].sort(key=lambda e:(e.get('landcover',{}).get('bare_sparse_fraction',0)<.5,e['distance']))
        t['catalog_order_note']='Within fixed terrain-nearest shortlist, bare/sparse>=50% shown first; not a global landcover-constrained optimum'
    path.write_text(json.dumps(m,ensure_ascii=False,indent=2))
    guide='''# Mars click MVP\n\nCopy mars_types.json, mars_types_grid.bin and mars_analogs.mjs to the same public folder.\n\n```js\nimport {loadMarsCatalog} from './mars_analogs.mjs';\nconst mars = await loadMarsCatalog('/mars/');\nconst result = mars.lookup(18.44, 77.45);\nconsole.log(result.type.label, result.earthCandidates, result.climate);\n```\n\nLatitude is degrees north; longitude degrees east, accepts -180..180 or 0..360.\nFor your 3D hit point use mesh-local coordinates and verify globe rotation/texture conventions.\nThe optional pointToLatLon assumes +Y north, +X longitude 0, +Z longitude 90E.\nValidate Jezero (18.44,77.45), both poles and the dateline on YOUR globe.\n\nNo ML or external APIs run on click. 10 types are regional terrain prototypes; map lookup step 0.5 degrees.\nCluster median features are NOT exact point features. Every point in a type returns its fixed catalog.\nMars climate is a separate coarse MCD model snapshot: Ls=0, noon local time, average-solar climatology; atmospheric fields at 10m.\nEarth POWER is long-term climatology (air temperature at 2m): climate context is NOT matched or scored across planets.\nMissing Mars humidity and precipitation are null, never zero. No material classes inferred from a DEM.\nEarth cover comes from 2021; sites are regional candidates, not surveyed/approved testing grounds.\nLunar data/model are NOT included; return unsupported for Moon until a lunar dataset is prepared.\n'''
    (public/'MARS_INTEGRATION.md').write_text(guide)
    with zipfile.ZipFile(public/'mars_demo_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['mars_types.json','mars_types_grid.bin','mars_analogs.mjs','MARS_INTEGRATION.md']:archive.write(public/name,name)
    (ROOT/'data/mars_types/package_checksums.json').write_text(json.dumps({name:sha256(public/name) for name in ['mars_types.json','mars_types_grid.bin','mars_analogs.mjs']},indent=2))
    print('Package prepared', (public/'mars_demo_bundle.zip').stat().st_size,'bytes')

if __name__=='__main__':main()
