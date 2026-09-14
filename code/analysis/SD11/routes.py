from pathlib import Path
import os,sys,json,ast,math
from types import SimpleNamespace
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_process/runtime_support/route_scope_vendor'))
import numpy as np
import pandas as pd
import searoute as sr
from shapely.geometry import LineString,box
from historical import inputs,h,CAP
OUT=HERE/'results/routes';OUT.mkdir(parents=True,exist_ok=True)

def main():
    old=ROOT/'research_process/nickel_route_extension_20260908'
    template=old/'frozen/reproduction/project_snapshot/route_template/inputs'
    ns=dict(math=math,LineString=LineString,box=box)
    tree=ast.parse((old/'audit_geometric_incidence.py').read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='robust_labels')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'retained_geometry_detector','exec'),ns)
    boxes=[SimpleNamespace(**r) for r in pd.read_csv(template/'chokepoint_boxes.csv').to_dict('records')]
    coords={r.iso3:(r.lon,r.lat) for r in pd.read_csv(old/'results/country_anchor_audit.csv').itertuples() if r.status=='available'}
    coords.update({r.iso3:(r.lon,r.lat) for r in pd.read_csv(template/'representative_ports.csv').itertuples()})
    pairs=pd.read_csv(CAP/'outputs/routes_rebuilt/pair_audit.csv').fillna({'labels':''})
    annual,windows,c,blocks,pair,chokes=inputs()
    baseline,cons=h.recompute(blocks,len(annual));assert cons<1e-10
    assert np.nanmax(abs(baseline[:,5]-annual.top_chokepoint_share))<1e-10
    summaries=[];audits=[];members=[]
    basecat=h.classify(baseline,pair)
    adjacent=windows.window_family.eq('adjacent').to_numpy()
    for closed,passage in [('Panama','panama'),('Suez','suez'),('Malacca','malacca')]:
        cache=OUT/(closed+'_alternative_paths.json')
        paths=json.loads(cache.read_text()) if cache.exists() else {}
        affected=pairs[(pairs['mode']=='sea')&pairs.labels.str.split('|',regex=False).map(lambda x:closed in x)]
        print(closed,'affected pairs',len(affected),flush=True)
        for k,r in enumerate(affected.itertuples()):
            token=r.origin+'|'+r.dest
            if token not in paths:
                try:
                    route=sr.searoute(coords[r.origin],coords[r.dest],units='km',speed_knot=24,append_orig_dest=False,restrictions=['northwest',passage],include_ports=False,port_params={},return_passages=False,algorithm=None,backend='networkx')
                    vertices=route.geometry['coordinates'];labels=ns['robust_labels'](vertices,boxes)
                    assert len(vertices)>=2 and np.isfinite(vertices).all() and closed not in labels
                    paths[token]=dict(labels=labels,vertices=vertices,length_km=route.properties['length'],baseline_length_km=r.length_km,accepted=True)
                except Exception as e:paths[token]=dict(accepted=False,error=str(e))
            if k%200==0:cache.write_text(json.dumps(paths));print(closed,k,flush=True)
        cache.write_text(json.dumps(paths))
        failed={k:v for k,v in paths.items() if not v['accepted']}
        (OUT/(closed+'_failures.json')).write_text(json.dumps(failed,indent=2))
        changed=0;covered=0.;unresolved=0.;total=float(c.value.sum())
        # Recalculate only route shares. Trade, origins and cohort remain identical.
        variant=baseline.copy();upper=baseline.copy();top=np.full(len(annual),-1,dtype=int)
        for b in blocks:
            cs=b['countries'];n=b['n']
            for s in b['stages']:
                inc=s['incidence'].copy();inc_upper=inc.copy()
                for j,(e,i) in enumerate(zip(cs[s['exp']],cs[s['imp']])):
                    alt=paths.get(e+'|'+i)
                    if alt is not None:
                        if alt['accepted']:
                            inc[j]=[x in alt['labels'] for x in chokes];inc_upper[j]=inc[j]
                        else:
                            # Failed routes remain in the denominator, with unknown incidence.
                            inc[j]=0.;inc_upper[j]=1.;unresolved+=float(s['value'][j])
                        covered+=float(s['value'][j]);changed+=1
                values=np.zeros((n,len(chokes)));np.add.at(values,s['imp'],s['value'][:,None]*inc)
                denominator=np.bincount(s['imp'],weights=s['value'],minlength=n)
                variant[s['unit'],5]=values.max(axis=1)[s['outputs']]/denominator[s['outputs']]
                top[s['unit']]=values.argmax(axis=1)[s['outputs']]
                uvalues=np.zeros((n,len(chokes)));np.add.at(uvalues,s['imp'],s['value'][:,None]*inc_upper)
                upper[s['unit'],5]=uvalues.max(axis=1)[s['outputs']]/denominator[s['outputs']]
        cats=h.classify(variant,pair);delta=variant[pair[1]]-variant[pair[0]]
        dl=variant[pair[1],5]-upper[pair[0],5]
        du=upper[pair[1],5]-variant[pair[0],5]
        event=adjacent&cats['apparent']&(dl>=.025)
        possible=adjacent&cats['apparent']&(du>=.025)
        frozen=adjacent&basecat['apparent']&((baseline[pair[1],5]-baseline[pair[0],5])>=.025)
        for year in ['pooled']+list(range(2013,2025)):
            select=adjacent if year=='pooled' else adjacent&windows.end_year.eq(year).to_numpy()
            apparent=select&cats['apparent'];w=cats['value'];den=w[apparent].sum()
            ev=select&event;keep=top[pair[0]]==top[pair[1]]
            transfer_lower=apparent&((delta[:,2]>=.025)|(dl>=.025));transfer_upper=apparent&((delta[:,2]>=.025)|(du>=.025))
            summaries.append(dict(scenario='avoid_'+closed,end_year=year,apparent_count=int(apparent.sum()),route_events_lower=int(ev.sum()),route_events_upper=int((select&possible).sum()),transfer_lower_pct=100*w[transfer_lower].sum()/den,transfer_upper_pct=100*w[transfer_upper].sum()/den,route_event_lower_pct=100*w[ev].sum()/den,route_event_upper_pct=100*w[select&possible].sum()/den,unresolved_pair_count=len(failed)))
        d=windows.loc[adjacent,['importer_iso','metal','stage','baseline_year','end_year']].copy()
        d['scenario']='avoid_'+closed;d['delta_route_lower']=dl[adjacent];d['delta_route_upper']=du[adjacent];d['route_event_lower']=event[adjacent];d['route_event_upper']=possible[adjacent];members.append(d)
        for key,v in paths.items():
            if v['accepted']:audits.append(dict(scenario='avoid_'+closed,pair=key,baseline_length_km=v['baseline_length_km'],alternative_length_km=v['length_km'],distance_ratio=v['length_km']/v['baseline_length_km'],labels='|'.join(v['labels'])))
        (OUT/(closed+'_coverage.json')).write_text(json.dumps(dict(affected_pairs=len(paths),unresolved_pairs=len(failed),changed_corridor_years=changed,affected_value_share=covered/total,unresolved_value_share=unresolved/total,trade_value_usd=total,mode_classification_fixed=True),indent=2))
        print(closed,'complete coverage',covered/total,flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'alternative_path_audit.csv',index=False)
    pd.concat(members).to_csv(OUT/'event_membership.csv.gz',index=False)
    (OUT/'QA.json').write_text(json.dumps(dict(accepted=True,scope='Single-passage alternative-path stress with conservative failed-path bounds, not historical routes or empirical validation',land_classification_fixed=True,scenario_count=3),indent=2))
if __name__=='__main__':main()
