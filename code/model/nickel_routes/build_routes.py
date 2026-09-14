"""Extend the frozen country-pair routing model with explicit provenance."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = ROOT/'research_process/runtime_support/route_scope_vendor'
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT/'research_process/nickel_scope_correction_20260908'))
import run_scope_reanalysis as scope
import pandas as pd
import numpy as np
import networkx as nx
import searoute as sr
import shapefile
import shapely
from shapely.geometry import Point, shape, LineString, box
from shapely.ops import unary_union

OUT = HERE/'results'
FROZEN = HERE/'frozen'
OLD = ROOT/'research_process/numerical_checks/recalculation_tables/routes/minerals_lanes_geography_corrected.csv'
AUDIT = ROOT/'research_process/numerical_checks/recalculation_tables/routes/country_geography_audit.csv'
GEODIST = ROOT/'external_sources/cepii_geodist/geo_cepii.dta'
COUNTRY_CODES = ROOT/'external_sources/baci_hs07_202601/country_codes_V202601.csv'
SCOPE = ROOT/'research_process/nickel_scope_correction_20260908/results'


def sha(path):
    return scope.sha(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(scope.ARCHIVE) as z:
        members = [n for n in z.namelist() if n.startswith('reproduction/project_snapshot/route_template/') and (n.endswith('.csv') or n.endswith('.json') or n.endswith('generate_route_template.py'))]
        members += [n for n in z.namelist() if n.startswith('reproduction/revision/frozen_inputs/natural_earth/')]
        hashes = {}
        for name in members:
            data = z.read(name)
            path = (FROZEN/name).resolve()
            assert path.is_relative_to(FROZEN.resolve()) and len(data)<20_000_000
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == data
            else:
                path.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
    template = FROZEN/'reproduction/project_snapshot/route_template'
    rt = scope.load_module('route_extension_generator', template/'generate_route_template.py')
    assets = Path(sr.__file__).parent
    assert sha(assets/'searoute.py') == rt.EXPECTED_SEAROUTE_MODULE_SHA256
    assert sha(assets/'data/marnet_dict.py') == rt.EXPECTED_MARNET_SHA256
    for filename, label in [('nation_lat_lon.csv','country coordinates'), ('representative_ports.csv','representative ports'), ('chokepoint_boxes.csv','chokepoint boxes')]:
        rt.verify_input(template/'inputs'/filename, label)
    (OUT/'frozen_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    return rt, template, assets


def anchors(rt, template, required):
    old = pd.read_csv(AUDIT).set_index('iso3')
    geo = pd.read_stata(GEODIST)
    codes = pd.read_csv(COUNTRY_CODES).dropna(subset=['country_iso2','country_iso3'])
    counts = codes.groupby('country_iso2').country_iso3.nunique()
    mapping = codes[codes.country_iso2.isin(counts[counts.eq(1)].index)].drop_duplicates('country_iso2').set_index('country_iso2').country_iso3
    # CEPII ROM/PAL and BACI ROU/PSE retain identical unique ISO2 identifiers.
    # No fuzzy country-name matching or territorial backfilling is used.
    geo['source_iso3'] = geo.iso3
    geo['iso3'] = geo.iso2.map(mapping).fillna(geo.iso3)
    geo = geo[geo.maincity.eq(1)].set_index('iso3')
    assert not geo.index.duplicated().any()
    reader = shapefile.Reader(str(FROZEN/'reproduction/revision/frozen_inputs/natural_earth/ne_110m_admin_0_countries.shp'), encoding='utf-8')
    country_shapes = {r.record.as_dict()['ADM0_A3']: shape(r.shape.__geo_interface__) for r in reader.iterShapeRecords()}
    union = unary_union(list(country_shapes.values()))
    components = list(union.geoms)
    coords, part, records = {}, {}, []
    for iso in sorted(required | set(old.index)):
        if iso in old.index:
            r = old.loc[iso]
            pt = Point(r.new_lon, r.new_lat)
            source = 'current_corrected_country_audit'
        elif iso in geo.index:
            r = geo.loc[iso]
            pt = Point(float(r.lon), float(r.lat))
            source = 'CEPII_GeoDist_maincity1'
        else:
            records.append(dict(iso3=iso, status='missing_coordinate'))
            continue
        geom = country_shapes.get(iso)
        replaced = False
        if geom is not None and geom.distance(pt)>1.:
            largest = max(geom.geoms, key=lambda x:x.area) if hasattr(geom,'geoms') else geom
            pt = largest.representative_point()
            replaced = True
        component = min(range(len(components)), key=lambda i:components[i].distance(pt))
        gap = float(components[component].distance(pt))
        part[iso] = component if geom is not None and gap<.5 else None
        coords[iso] = (float(pt.x),float(pt.y))
        records.append(dict(iso3=iso,status='available',lon=pt.x,lat=pt.y,anchor_source=source,polygon_replacement=replaced,
            closest_land_gap_degrees=gap,land_component_before_links=part[iso],geodist_city=geo.loc[iso,'city_en'] if iso in geo.index else '',
            geodist_source_iso3=geo.loc[iso,'source_iso3'] if iso in geo.index else ''))
    for a,b in [('GBR','FRA'),('DNK','DEU'),('DNK','SWE'),('BHR','SAU'),('SGP','MYS')]:
        previous, target = part.get(a), part.get(b)
        assert target is not None
        if previous is not None:
            for iso in part:
                if part[iso] == previous:
                    part[iso] = target
        part[a] = target
    for r in records:
        r['land_component_with_fixed_links'] = part.get(r['iso3'])
    # Verify the old country-anchor land partition has not been silently changed.
    common = [iso for iso in old.index if pd.notna(old.at[iso,'land_component_with_fixed_links'])]
    assert all(part.get(iso) is not None for iso in common)
    for a in common:
        for b in common:
            assert (part[a]==part[b]) == (old.at[a,'land_component_with_fixed_links']==old.at[b,'land_component_with_fixed_links'])
    pd.DataFrame(records).to_csv(OUT/'country_anchor_audit.csv',index=False)
    return coords, part, rt.load_ports(template/'inputs/representative_ports.csv')


def required_pairs():
    pairs = set()
    kinds = {}
    # Direct observations include all annual diagnostic units, not just optimized ones.
    for variant in ('original3','expanded6'):
        trade = pd.read_csv(SCOPE/variant/'trade_corridors.csv.gz')
        annual = pd.read_csv(SCOPE/variant/'annual_indicators.csv.gz')
        for pair in zip(trade.exporter_iso,trade.importer_iso):
            pairs.add(pair)
            kinds.setdefault(pair,set()).add('positive_trade_corridor')
        for year in range(2012,2025):
            current = trade[trade.year.eq(year)]
            early = annual[annual.year.eq(year-4)]
            end = annual[annual.year.eq(year)]
            units = end[end.direct_total_value_usd.ge(1e7)].merge(early[early.direct_total_value_usd.ge(1e6)][scope.KEYS],on=scope.KEYS)
            history = trade[trade.year.between(year-5,year)].groupby(['stage','exporter_iso','year']).reconstructed_value_usd.sum().groupby(['stage','exporter_iso']).max()
            active = current.groupby(['stage','exporter_iso']).reconstructed_value_usd.sum()
            for stage, group in units.groupby('stage'):
                peaks = history.loc[stage]
                peaks = peaks[peaks.index.isin(active.loc[stage].index)].sort_values(ascending=False).head(60)
                for unit in group.itertuples():
                    observed = current[current.stage.eq(stage)&current.importer_iso.eq(unit.importer_iso)].exporter_iso
                    for exporter in set(peaks.index)|set(observed):
                        if exporter == unit.importer_iso:
                            continue
                        pair = (str(exporter),str(unit.importer_iso))
                        pairs.add(pair)
                        kinds.setdefault(pair,set()).add('current_year_candidate_supplier')
    pd.DataFrame([dict(origin=a,dest=b,requirement='|'.join(sorted(kinds[(a,b)]))) for a,b in sorted(pairs)]).to_csv(OUT/'required_country_pairs.csv',index=False)
    return pairs


def main():
    rt, template, assets = prepare()
    old = pd.read_csv(OLD).fillna({'chokepoints':''})
    pairs = old[['origin','dest','mode','chokepoints']].drop_duplicates()
    assert not pairs.duplicated(['origin','dest']).any()
    existing = {(r.origin,r.dest):(r.mode,r.chokepoints) for r in pairs.itertuples()}
    nickel = old[old.metal.eq('Nickel')].copy()
    nickel_keys = set(zip(nickel.origin,nickel.dest))
    needed = required_pairs()
    required_countries = set(x for p in needed for x in p)
    coords, part, ports = anchors(rt,template,required_countries)
    boxes = rt.load_chokepoints(template/'inputs/chokepoint_boxes.csv')
    additions, audit, paths = [], [], {}
    for index,(a,b) in enumerate(sorted(needed)):
        key = (a,b)
        if key in nickel_keys:
            audit.append(dict(origin=a,dest=b,status='retained_nickel_template',mode=existing[key][0],chokepoints=existing[key][1]))
            continue
        if key in existing:
            mode, labels = existing[key]
            audit.append(dict(origin=a,dest=b,status='shared_static_pair',mode=mode,chokepoints=labels))
        elif a not in coords or b not in coords:
            audit.append(dict(origin=a,dest=b,status='unresolved_coordinate',missing_country='|'.join(x for x in key if x not in coords)))
            continue
        else:
            land = part.get(a) is not None and part.get(a)==part.get(b) and rt.haversine_km(coords[a],coords[b])<1500.
            if land:
                mode, labels = 'overland',''
                audit.append(dict(origin=a,dest=b,status='new_static_land',mode=mode,chokepoints=labels))
            else:
                start,end = ports.get(a,coords[a]),ports.get(b,coords[b])
                try:
                    result = sr.searoute(start,end,units='km',speed_knot=24,append_orig_dest=False,
                        restrictions=[rt.Passage.northwest],include_ports=False,port_params={},return_passages=False,algorithm=None,backend='networkx')
                    vertices = result.geometry['coordinates']
                    assert len(vertices)>=2
                    assert np.isfinite(np.asarray(vertices)).all()
                    mode, labels = 'sea','|'.join(rt.chokepoint_labels(vertices,boxes))
                    paths[f'{a}|{b}'] = vertices
                    audit.append(dict(origin=a,dest=b,status='new_static_sea',mode=mode,chokepoints=labels,
                        origin_lon=start[0],origin_lat=start[1],dest_lon=end[0],dest_lat=end[1],origin_port_override=a in ports,dest_port_override=b in ports,
                        origin_network_displacement_km=rt.haversine_km(start,vertices[0]),dest_network_displacement_km=rt.haversine_km(end,vertices[-1]),route_length_km=result.properties['length']))
                except Exception as exc:
                    audit.append(dict(origin=a,dest=b,status='unresolved_route',reason=f'{type(exc).__name__}: {exc}'))
                    continue
        additions.append(dict(metal='Nickel',origin=a,dest=b,value_usd=0,mode=mode,chokepoints=labels))
        if index%100==0:
            print(f'pairs {index}/{len(needed)}; added {len(additions)}; new sea geometries {len(paths)}',flush=True)
    augmented = pd.concat([old,pd.DataFrame(additions)],ignore_index=True)
    assert not augmented.duplicated(['metal','origin','dest']).any()
    # Existing rows are immutable, including the unused legacy template value column.
    pd.testing.assert_frame_equal(augmented.iloc[:len(old)].reset_index(drop=True),old.reset_index(drop=True),check_dtype=False)
    augmented.to_csv(OUT/'minerals_lanes_nickel_extended.csv',index=False)
    table = pd.DataFrame(audit)
    table.to_csv(OUT/'route_pair_provenance.csv',index=False)
    (OUT/'generated_route_geometries.json').write_text(json.dumps(paths),encoding='utf-8')
    manifest = dict(status='static_geometry_extension_not_observed_shipping',required_pairs=len(needed),added_nickel_pairs=len(additions),
        route_rows=len(augmented),pair_status=table.status.value_counts().to_dict(),
        python=sys.version,pandas=pd.__version__,networkx=nx.__version__,shapely=shapely.__version__,searoute='1.6.0',
        route_module_sha256=sha(assets/'searoute.py'),network_sha256=sha(assets/'data/marnet_dict.py'),
        inputs={str(p):sha(p) for p in (OLD,AUDIT,GEODIST,COUNTRY_CODES)},code_sha256=sha(__file__),
        value_column='Zero for added route definitions; ignored by indicators and optimization; real annual trade values come from BACI.',
        candidate_support='Current-year positive suppliers and rolling six-year peak ranks; geometric support only is reused across years.')
    (OUT/'route_extension_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
