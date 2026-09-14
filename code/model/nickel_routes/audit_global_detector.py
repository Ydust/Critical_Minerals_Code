"""All-mineral detector repair on existing support, isolating the geometry error."""
from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd

from build_routes import OUT, ROOT, OLD, prepare, sr, scope, sha
from audit_geometric_incidence import robust_labels
from validate_and_summarize import independent_labels


def main():
    dest=OUT/'global_detector_audit'
    dest.mkdir(parents=True,exist_ok=True)
    rt,template,_=prepare()
    boxes=rt.load_chokepoints(template/'inputs/chokepoint_boxes.csv')
    ports=rt.load_ports(template/'inputs/representative_ports.csv')
    anchors=pd.read_csv(OUT/'country_anchor_audit.csv').query('status == "available"').set_index('iso3')
    coords={iso:(r.lon,r.lat) for iso,r in anchors.iterrows()}
    paths=json.loads((OUT/'all_nickel_route_geometries.json').read_text())
    old=pd.read_csv(OLD).fillna({'chokepoints':''})
    pairs=old[['origin','dest','mode','chokepoints']].drop_duplicates()
    assert not pairs.duplicated(['origin','dest']).any()
    rows=[]
    additional={}
    for i,r in enumerate(pairs.itertuples()):
        key=f'{r.origin}|{r.dest}'
        status='resolved'
        if r.mode=='overland':
            assert r.chokepoints==''
            legacy,corrected='',''
        else:
            assert r.mode=='sea'
            vertices=paths.get(key)
            if vertices is None:
                start,end=ports.get(r.origin,coords.get(r.origin)),ports.get(r.dest,coords.get(r.dest))
                assert start is not None and end is not None,key
                result=sr.searoute(start,end,units='km',speed_knot=24,append_orig_dest=False,restrictions=[rt.Passage.northwest],
                    include_ports=False,port_params={},return_passages=False,algorithm=None,backend='networkx')
                vertices=result.geometry['coordinates']
                additional[key]=vertices
            legacy='|'.join(rt.chokepoint_labels(vertices,boxes))
            if len(vertices)<2:
                status='unresolved_collapsed_geometry'
                corrected=None
            else:
                corrected='|'.join(robust_labels(vertices,boxes))
                assert corrected=='|'.join(independent_labels(vertices,boxes)),key
            assert legacy==r.chokepoints,(key,legacy,r.chokepoints)
        rows.append(dict(origin=r.origin,dest=r.dest,mode=r.mode,old_labels=r.chokepoints,corrected_labels=corrected,status=status))
        if i%500==0:
            print(f'global existing pairs {i}/{len(pairs)}',flush=True)
    audit=pd.DataFrame(rows)
    audit['incidence_changed']=audit.status.eq('resolved')&audit.old_labels.ne(audit.corrected_labels)
    audit.to_csv(dest/'country_pair_detector_audit.csv',index=False)
    (dest/'additional_route_geometries.json').write_text(json.dumps(additional),encoding='utf-8')
    changed=old.merge(audit,on=['origin','dest','mode'],validate='many_to_one')
    changed[changed.incidence_changed|changed.status.ne('resolved')].to_csv(dest/'affected_mineral_route_rows.csv',index=False)
    corrected=changed[changed.status.eq('resolved')][['metal','origin','dest','value_usd','mode','corrected_labels']].rename(columns={'corrected_labels':'chokepoints'})
    path=dest/'minerals_lanes_existing_support_geometric.csv'
    corrected.to_csv(path,index=False)
    # No extra nickel products or candidate routes enter this all-mineral control.
    sys.path.insert(0,str(ROOT/'research_process/evidence_extension_20260908'))
    import run_historical_joint_sensitivity as hs
    annual,corridor,seed,lanes,windows,hashes,records=hs.load_inputs()
    blocks,pair,_,_=hs.prepare(annual,corridor,seed,lanes,windows)
    baseline,error=hs.recompute(blocks,len(annual))
    oldmatrix=annual[hs.FIELDS].to_numpy()
    assert np.allclose(baseline,oldmatrix,rtol=1e-12,atol=1e-12,equal_nan=True)
    oldcats=hs.classify(baseline,pair)
    apparent=(windows.three_layer_evidence_eligible&windows.apparent_derisking).to_numpy()
    assert np.array_equal(oldcats['transfer'],apparent&windows.classification.isin(['mine_origin_transfer','route_transfer','multiple_risk_transfer']).to_numpy())
    assert np.array_equal(oldcats['substantive'],apparent&windows.classification.eq('substantive_derisking').to_numpy())
    blocks,newpair,_,_=hs.prepare(annual,corridor,seed,corrected,windows)
    new,error=hs.recompute(blocks,len(annual))
    assert np.array_equal(newpair[0],pair[0]) and np.array_equal(newpair[1],pair[1])
    assert np.allclose(new[:,:4],baseline[:,:4],rtol=0,atol=0,equal_nan=True)
    newcats=hs.classify(new,pair)
    frames=[]
    for name,cats in [('existing_detector',oldcats),('periodic_segment_detector',newcats)]:
        frame=pd.DataFrame(hs.summarize(cats,list(hs.selections(windows)),0,-1.))
        frame.insert(0,'detector',name)
        frames.append(frame)
    summary=pd.concat(frames,ignore_index=True)
    summary.to_csv(dest/'historical_detector_comparison.csv',index=False)
    output=annual[hs.KEY].copy()
    for j,col in enumerate(hs.FIELDS):
        output[col]=new[:,j]
        output[col+'_previous']=baseline[:,j]
    output.to_csv(dest/'annual_detector_comparison.csv.gz',index=False)
    windows_out=windows[['importer_iso','metal','stage','window_family','baseline_year','end_year']].copy()
    for col in ['eligible','apparent','transfer','substantive']:
        windows_out[col+'_previous']=oldcats[col]
        windows_out[col+'_corrected']=newcats[col]
    windows_out.to_csv(dest/'window_classification_changes.csv.gz',index=False)
    manifest=dict(country_pairs=len(pairs),incidence_changed_pairs=int(audit.incidence_changed.sum()),
        changed_mineral_route_rows=int(changed.incidence_changed.sum()),collapsed_country_pairs=int(audit.status.ne('resolved').sum()),
        removed_collapsed_mineral_rows=int(changed.status.ne('resolved').sum()),regenerated_legacy_disagreements=0,
        unit_years=len(annual),comparison_windows=len(windows),supported_minerals=int(annual.loc[annual.origin_hhi.notna(),'metal'].nunique()),
        changed_annual_chokepoint_values=int((np.abs(new[:,5]-baseline[:,5])>1e-12).sum()),
        changed_annual_coverage=int((np.abs(new[:,4]-baseline[:,4])>1e-12).sum()),
        code_sha256=sha(__file__),source_hashes=hashes,
        boundary='Existing product scope, trade reconstruction, origin profiles and quality index retained. Static detector repair only, not a final expanded-product manuscript rerun.')
    (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
    print(summary[summary.metal.eq('all')&summary.end_year.eq('pooled')][['detector','window_family','apparent_count','transfer_share','substantive_share']].to_string(index=False))


if __name__=='__main__':
    main()
