"""Recompute static sea paths; do not imply observed-voyage validation."""
from pathlib import Path
import ast, hashlib, json, math, sys, time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research_process/nickel_route_extension_20260908'
VENDOR=ROOT/'research_process/runtime_support/route_scope_vendor'
sys.path.insert(0,str(VENDOR))
import numpy as np
import pandas as pd
import networkx as nx
import searoute as sr
from searoute.classes.passages import Passage
from shapely.geometry import LineString, box

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

def main():
    out=HERE/'capsule/outputs/routes_rebuilt';out.mkdir(parents=True,exist_ok=True)
    template=OLD/'frozen/reproduction/project_snapshot/route_template/inputs'
    engine=Path(sr.__file__).parent
    locks={engine/'searoute.py':'4ab006ba097e2d01807ccbe6e215756ccab5338920bb47042db9599b467dcaf0',engine/'data/marnet_dict.py':'3622fff283e71b940ba5d58280b07c5cb2e95959ffb48f647b39b500b6c50ea3',template/'representative_ports.csv':'d7e8409f18afa7cf64ae9d88f2081b6bdbb7caae4e08c0df2b96b32686422509',template/'chokepoint_boxes.csv':'d24df68e7e7361ae6f3829a768f0d1fe91b2b5f5d8078a67ddf9d1b7e979c787'}
    for p,h in locks.items(): assert sha(p)==h,str(p)
    assert nx.__version__=='3.5'
    namespace=dict(math=math,LineString=LineString,box=box)
    extract(OLD/'audit_geometric_incidence.py',['robust_labels'],namespace)
    extract(OLD/'validate_and_summarize.py',['clips_rectangle','independent_labels'],namespace)
    robust=namespace['robust_labels'];independent=namespace['independent_labels']
    boxes=[SimpleNamespace(**r) for r in pd.read_csv(template/'chokepoint_boxes.csv').to_dict('records')]
    assert 'Panama' in independent([[280.5,8.6],[280,9.75]],boxes)
    assert 'Panama' not in independent([[179,9],[-179,9]],boxes)
    assert 'Gibraltar' in independent([[-7,36],[-4,36]],boxes)
    coords={r.iso3:(r.lon,r.lat) for r in pd.read_csv(OLD/'results/country_anchor_audit.csv').itertuples() if r.status=='available'}
    ports={r.iso3:(r.lon,r.lat) for r in pd.read_csv(template/'representative_ports.csv').itertuples()}
    routes=pd.read_csv(HERE/'capsule/inputs/historical/routes.csv').fillna({'chokepoints':''})
    pairs=routes[['origin','dest','mode','chokepoints']].drop_duplicates()
    assert not pairs.duplicated(['origin','dest']).any()
    rows=[];paths={};start=time.monotonic()
    for i,r in enumerate(pairs.itertuples()):
        key=f'{r.origin}|{r.dest}'
        if r.mode=='overland':
            assert r.chokepoints==''
            rows.append(dict(origin=r.origin,dest=r.dest,mode=r.mode,labels='',matches=True,scope='retained audited land classification'))
            continue
        assert r.mode=='sea'
        a=ports.get(r.origin,coords.get(r.origin));b=ports.get(r.dest,coords.get(r.dest))
        assert a is not None and b is not None,key
        result=sr.searoute(a,b,units='km',speed_knot=24,append_orig_dest=False,restrictions=[Passage.northwest],include_ports=False,port_params={},return_passages=False,algorithm=None,backend='networkx')
        vertices=result.geometry['coordinates'];assert len(vertices)>=2 and np.isfinite(vertices).all(),key
        labels='|'.join(robust(vertices,boxes));assert labels=='|'.join(independent(vertices,boxes)),key
        paths[key]=vertices
        rows.append(dict(origin=r.origin,dest=r.dest,mode=r.mode,labels=labels,matches=labels==r.chokepoints,scope='fresh locked-network sea path',length_km=result.properties['length']))
        if i%500==0: print(f'route {i}/{len(pairs)}',flush=True)
    audit=pd.DataFrame(rows);audit.to_csv(out/'pair_audit.csv',index=False)
    (out/'sea_geometries.json').write_text(json.dumps(paths),encoding='utf-8')
    rebuilt=routes.drop(columns='chokepoints').merge(audit[['origin','dest','labels']],on=['origin','dest'],validate='many_to_one').rename(columns={'labels':'chokepoints'})
    rebuilt.to_csv(out/'routes.csv',index=False)
    inputs=list(locks)+[OLD/'results/country_anchor_audit.csv',HERE/'capsule/inputs/historical/routes.csv',OLD/'audit_geometric_incidence.py',OLD/'validate_and_summarize.py',__file__]
    manifest={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in inputs}
    runtime={str(p.relative_to(VENDOR)):sha(p) for p in VENDOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (out/'INPUT_AND_RUNTIME_MANIFEST.json').write_text(json.dumps(dict(inputs=manifest,route_vendor=runtime),indent=2),encoding='utf-8')
    report=dict(accepted=bool(audit.matches.all()),route_rows=len(rebuilt),country_pairs=len(audit),fresh_sea_paths=len(paths),retained_land_pairs=int(audit.mode.eq('overland').sum()) if False else int((audit['mode']=='overland').sum()),incidence_mismatches=int((~audit.matches).sum()),independent_detectors_agree=True,python=sys.version,numpy=np.__version__,pandas=pd.__version__,networkx=nx.__version__,searoute='1.6.0',runtime_origin='checksum-locked retained route vendor; not a fresh package install',elapsed_seconds=time.monotonic()-start,scope='Static-path reproducibility conditional on audited anchors and land classification. Not observed shipping validation. No new country-pair support.')
    (out/'ROUTE_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));assert report['accepted']

if __name__=='__main__':main()
