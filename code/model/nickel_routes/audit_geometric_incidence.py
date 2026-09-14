"""Correct periodic longitude and segment-crossing incidence, retaining controls."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from build_routes import OUT, FROZEN, OLD, ROOT, prepare, sr, sha
from shapely.geometry import LineString, box


def wrapped_vertices(vertices):
    return [[(float(x)+180.)%360.-180.,float(y)] for x,y in vertices]


def robust_labels(vertices, boxes):
    """Intersect short periodic segments, not world-spanning dateline chords."""
    selected = []
    for choke in boxes:
        hit = False
        for a,b in zip(vertices[:-1],vertices[1:]):
            x0,y0 = float(a[0]),float(a[1])
            x1 = x0 + (float(b[0])-x0+180.)%360.-180.
            y1 = float(b[1])
            lo,hi = sorted((x0,x1))
            k0 = math.ceil((lo-choke.xmax)/360.)
            k1 = math.floor((hi-choke.xmin)/360.)
            if min(y0,y1)>choke.ymax or max(y0,y1)<choke.ymin:
                continue
            segment = LineString([(x0,y0),(x1,y1)])
            for k in range(k0,k1+1):
                if segment.intersects(box(choke.xmin+360*k,choke.ymin,choke.xmax+360*k,choke.ymax)):
                    hit = True
                    break
            if hit:
                break
        if hit:
            selected.append(choke.label)
    return selected


def main():
    rt,template,assets = prepare()
    inputs = template/'inputs'
    ports = rt.load_ports(inputs/'representative_ports.csv')
    anchors = pd.read_csv(OUT/'country_anchor_audit.csv')
    anchors = anchors[anchors.status.eq('available')].set_index('iso3')
    coords = {iso:(r.lon,r.lat) for iso,r in anchors.iterrows()}
    boxes = rt.load_chokepoints(inputs/'chokepoint_boxes.csv')
    extended = pd.read_csv(OUT/'minerals_lanes_nickel_extended.csv').fillna({'chokepoints':''})
    nickel = extended[extended.metal.eq('Nickel')]
    existing_paths = json.loads((OUT/'generated_route_geometries.json').read_text())
    oldkeys = set(zip(pd.read_csv(OLD).query('metal == "Nickel"').origin,pd.read_csv(OLD).query('metal == "Nickel"').dest))
    records,paths = [],{}
    for i,row in enumerate(nickel.itertuples()):
        pair = f'{row.origin}|{row.dest}'
        if row.mode != 'sea':
            assert row.mode=='overland' and row.chokepoints==''
            records.append(dict(origin=row.origin,dest=row.dest,mode=row.mode,old_labels='',regenerated_legacy_labels='',wrapped_vertex_labels='',segment_labels='',existing_nickel_pair=(row.origin,row.dest) in oldkeys))
            continue
        vertices = existing_paths.get(pair)
        if vertices is None:
            start,end = ports.get(row.origin,coords.get(row.origin)),ports.get(row.dest,coords.get(row.dest))
            assert start is not None and end is not None, pair
            route = sr.searoute(start,end,units='km',speed_knot=24,append_orig_dest=False,
                restrictions=[rt.Passage.northwest],include_ports=False,port_params={},return_passages=False,algorithm=None,backend='networkx')
            vertices = route.geometry['coordinates']
        if len(vertices)<2:
            records.append(dict(origin=row.origin,dest=row.dest,mode=row.mode,old_labels=row.chokepoints,
                regenerated_legacy_labels='|'.join(rt.chokepoint_labels(vertices,boxes)),wrapped_vertex_labels=None,segment_labels=None,
                geometry_status='unresolved_collapsed_geometry',collapsed_vertices_json=json.dumps(vertices),
                existing_nickel_pair=(row.origin,row.dest) in oldkeys))
            continue
        legacy = rt.chokepoint_labels(vertices,boxes)
        wrapped = rt.chokepoint_labels(wrapped_vertices(vertices),boxes)
        corrected = robust_labels(vertices,boxes)
        assert set(legacy).issubset(corrected) and set(wrapped).issubset(corrected)
        records.append(dict(origin=row.origin,dest=row.dest,mode=row.mode,old_labels=row.chokepoints,
            regenerated_legacy_labels='|'.join(legacy),wrapped_vertex_labels='|'.join(wrapped),segment_labels='|'.join(corrected),
            existing_nickel_pair=(row.origin,row.dest) in oldkeys,geometry_status='resolved',
            longitudes_outside_canonical=any(abs(p[0])>180 for p in vertices),
            min_lon=min(p[0] for p in vertices),max_lon=max(p[0] for p in vertices)))
        paths[pair] = vertices
        if i%500==0:
            print(f'incidence audit {i}/{len(nickel)}',flush=True)
    audit = pd.DataFrame(records)
    audit['regenerated_legacy_disagrees'] = audit.old_labels.ne(audit.regenerated_legacy_labels)
    unresolved = audit.geometry_status.eq('unresolved_collapsed_geometry')
    audit['longitude_correction_changes'] = ~unresolved & audit.regenerated_legacy_labels.ne(audit.wrapped_vertex_labels)
    audit['segment_correction_changes'] = ~unresolved & audit.wrapped_vertex_labels.ne(audit.segment_labels)
    audit['any_incidence_change'] = ~unresolved & audit.old_labels.ne(audit.segment_labels)
    audit.to_csv(OUT/'geometric_incidence_audit.csv',index=False)
    lookup = audit.set_index(['origin','dest']).segment_labels
    corrected = extended.copy()
    dropped = []
    for index,row in corrected[corrected.metal.eq('Nickel')].iterrows():
        value = lookup.loc[(row.origin,row.dest)]
        if pd.isna(value):
            dropped.append(index)
        else:
            corrected.at[index,'chokepoints'] = value
    corrected = corrected.drop(index=dropped)
    corrected.to_csv(OUT/'minerals_lanes_nickel_extended_geometric.csv',index=False)
    (OUT/'all_nickel_route_geometries.json').write_text(json.dumps(paths),encoding='utf-8')
    manifest = dict(total_nickel_pairs=len(audit),sea_pairs=len(paths),unresolved_collapsed_pairs=len(dropped),
        regenerated_legacy_disagreements=int(audit.regenerated_legacy_disagrees.sum()),
        longitude_correction_pairs=int(audit.longitude_correction_changes.sum()),
        segment_correction_pairs=int(audit.segment_correction_changes.sum()),
        any_changed_pairs=int(audit.any_incidence_change.sum()),
        changed_preexisting_nickel_pairs=int((audit.any_incidence_change&audit.existing_nickel_pair).sum()),
        input_sha256=sha(OUT/'minerals_lanes_nickel_extended.csv'),code_sha256=sha(__file__),
        inference='Geometric incidence correction within a static shortest-path model, not observed transit.')
    (OUT/'geometric_incidence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    main()
