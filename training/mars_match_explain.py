"""Audit the existing prototype retrieval and export exact, frontend-ready explanations."""
import json
import numpy as np
import pandas as pd
import joblib
from scipy.spatial import cKDTree
from global_terrain_data import ROOT
from mars_types import xyz,FIELDS

LABELS=['Медианный уклон','p90 уклона','Относительный перепад высот']
UNITS=['°','°','м']

def main():
    path=ROOT/'public/mars_types.json';m=json.loads(path.read_text())
    artifact=joblib.load(ROOT/'models/mars_terrain_types.joblib')
    scaler,model,remap=artifact['scaler'],artifact['model'],artifact['remap']
    earth=dict(np.load(ROOT/'data/mars_types/earth_descriptors.npz'))
    valid=np.flatnonzero(earth['valid'])
    vectors=scaler.transform(np.log1p(earth['features'][valid]))
    points=cKDTree(xyz(earth['lat'][valid].astype(float),earth['lon'][valid].astype(float)))
    rows=[]
    for t in m['types']:
        old=int(np.flatnonzero(remap==t['id'])[0]);center=model.cluster_centers_[old]
        physical=np.expm1(scaler.inverse_transform(center.reshape(1,-1)))[0]
        distances=np.linalg.norm(vectors-center,axis=1)
        t['retrieval_reference_distances']={f'p{q}':float(np.percentile(distances,q)) for q in [10,50,90]}
        t['retrieval_prototype']=dict(zip(FIELDS,map(float,physical)))
        t['retrieval_prototype_note']='Inverse-transformed KMeans centroid; differs from descriptive class median.'
        for catalog_position,e in enumerate(t['earth_candidates'],1):
            chord,index=points.query(xyz(np.array([e['lat']]),np.array([e['lon']])) )
            index=int(index[0]);assert float(chord[0])<1e-7
            values=np.array([e['features'][f] for f in FIELDS],dtype='float32')
            np.testing.assert_allclose(values,earth['features'][valid[index]],atol=1e-6,rtol=0)
            z=scaler.transform(np.log1p(values.reshape(1,-1)))[0]
            delta=z-center;parts=delta.astype('float64')**2
            distance=float(np.sqrt(parts.sum()))
            if not np.isclose(distance,e['distance'],atol=2e-6):raise ValueError('Catalog/model distance mismatch')
            rank=int((distances<distances[index]).sum()+1)
            comparisons=[]
            for j,f in enumerate(FIELDS):
                comparisons.append({'feature':f,'label':LABELS[j],'unit':UNITS[j],
                    'mars_prototype_value':float(physical[j]),'earth_value':float(values[j]),
                    'relative_difference':float((values[j]-physical[j])/max(abs(float(physical[j])),1e-9)),
                    'signed_standardized_difference':float(delta[j]),'squared_distance_contribution':float(parts[j]),
                    'mismatch_share':float(parts[j]/parts.sum()) if parts.sum()>0 else 0})
                rows.append({'type_id':t['id'],'candidate_id':e['id'],'catalog_position':catalog_position,
                             'terrain_rank':rank,**comparisons[-1]})
            smallest=int(np.argmin(np.abs(delta)));largest=int(np.argmax(np.abs(delta)))
            text=(f"Наиболее близкий параметр: {LABELS[smallest].lower()}. Земля: {values[smallest]:.2f} {UNITS[smallest]}, "
                  f"прототип Марса: {physical[smallest]:.2f} {UNITS[smallest]}. "
                  f"Наибольший вклад в остаточное отличие даёт {LABELS[largest].lower()}. "
                  'Это объяснение геометрического сходства; свойства грунта и климат не подтверждены как совпадающие.')
            max_relative=max(abs(p['relative_difference']) for p in comparisons)
            if max_relative>=.2:
                text+=f' Частичное соответствие: максимальное относительное отличие признака {max_relative:.0%}.'
            e['match_explanation']={'distance':distance,'distance_squared':float(parts.sum()),
                'terrain_rank_before_diversity_and_cover':rank,'eligible_earth_locations':len(valid),
                'comparisons':comparisons,'text_ru':text,
                'largest_residual_feature':FIELDS[largest],
                'max_relative_feature_difference':max_relative,
                'visible_partial_match':max_relative>=.2,
                'partial_match_note':'20% is a descriptive display threshold, not an engineering acceptance limit',
                'catalog_position_note':'Display order may prioritize bare cover within the original three-site shortlist; E-number is original discovery order.'}
    m['matching_method']={'clustering':'MiniBatchKMeans, 10 Mars regional terrain prototypes',
        'retrieval':'Exact nearest-neighbor retrieval to each trained prototype in shared feature space',
        'fields':FIELDS,'transform':'log1p, then subtract training Mars median and divide by training Mars IQR',
        'training_median_log1p':scaler.center_.tolist(),'training_iqr_log1p':scaler.scale_.tolist(),
        'distance':'sqrt(sum_j((z_earth_j - centroid_type_j)^2))',
        'diversity':'Select three sequentially with spherical Earth separation >500 km, scanning up to 20000 nearest locations',
        'cover':'After terrain selection, show bare/sparse>=50% candidates first within the fixed shortlist',
        'scope':'Regional morphology, not material classification, climate equivalence or passability',
        'candidate_locations':len(valid),'explanation':'Exact additive squared-distance contributions; not SHAP or causal effects'}
    path.write_text(json.dumps(m,ensure_ascii=False,indent=2))
    pd.DataFrame(rows).to_csv(ROOT/'reports/global_terrain/mars_match_explanations.csv',index=False)
    print('Audited',len(rows)//3,'matches against',len(valid),'Earth locations; exported exact explanations.')

if __name__=='__main__':main()
