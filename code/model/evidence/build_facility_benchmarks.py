"""Archive public operator disclosures and extract origin-linked facility output.

No credentials or manuscript contents are sent. This is a facility-level external
benchmark, not a national export-origin validation or a new fitted model input.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile

from lxml import html
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'external/facility_supply'
PACKAGE = ROOT.parents[1] / 'manuscript_package_20260907_final'
SOURCES = {
    'nornickel_ar2021_operations': 'https://ar2021.nornickel.com/en/business-overview/operational-performance.html',
    'nornickel_ar2023_flow': 'https://ar2023.nornickel.com/en/business-overview/production-flow.html',
    'nornickel_ar2024_operations': 'https://ar2024.nornickel.com/en/business-overview/operational-performance.html',
    'glencore_raglan_chain': 'https://www.glencore.ca/en/raglan/what-we-do/our-mining-activity',
    'glencore_nickel_recycling': 'https://www.glencore.ca/en/what-we-do/nickel',
    'vale_voiseys_bay': 'https://valebasemetals.com/our-operations/voiseys-bay/',
}


def download(item):
    name, url = item
    request = urllib.request.Request(url, headers={'User-Agent': 'ResearchDataAudit/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
        resolved_url = response.url
    path = OUT / (name + '.html')
    path.write_bytes(raw)
    return dict(source_id=name, url=url, resolved_url=resolved_url, file=path.name,
                retrieved_utc=datetime.now(timezone.utc).isoformat(),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def clean(node):
    return re.sub(r'\s+', ' ', node.text_content()).strip()


def extract_table(raw):
    tree = html.fromstring(raw)
    for table in tree.xpath('//table'):
        rows = [[clean(c) for c in tr.xpath('./th|./td')] for tr in table.xpath('.//tr')]
        if not any('HARJAVALTA (FINLAND)' in ' '.join(r).upper() for r in rows):
            continue
        header = next((r for r in rows if r[:2] == ['Asset', '2012']), None)
        if header is None:
            continue
        years = [int(x) for x in header[1:]]
        assert years == list(range(2012, 2022)), header
        start = next(i for i, r in enumerate(rows) if 'HARJAVALTA (FINLAND)' in ' '.join(r).upper())
        output = []
        for j, (metal, unit) in enumerate([('Nickel', 't'), ('Copper', 't'), ('Palladium', 'koz'), ('Platinum', 'koz')]):
            total, own = rows[start + 1 + 2*j], rows[start + 2 + 2*j]
            assert total[0] == f'{metal}, {unit}', total
            assert 'own Russian feed' in own[0], own
            assert len(total) == len(own) == 11
            for year, a, b in zip(years, total[1:], own[1:]):
                a, b = int(a.replace(',', '')), int(b.replace(',', ''))
                assert 0 <= b <= a
                # Rounded output quantities, not chemical feed mass. Half a
                # displayed unit on each side gives a conservative rounding bound.
                s = b/a
                rounded_lower = max(0., b-.5)/(a+.5)
                output.append(dict(facility='Norilsk Nickel Harjavalta', processor_iso='FIN',
                    origin_iso='RUS', element=metal, year=year, unit=unit,
                    total_saleable_output=a, own_russian_feed_output=b,
                    other_feed_output=a-b, own_russian_feed_output_share=s,
                    russian_feed_output_share_lower_rounding_aware=rounded_lower,
                    russian_feed_hhi_lower_if_country_origin_equivalent=rounded_lower**2,
                    source_id='nornickel_ar2021_operations',
                    source_url=SOURCES['nornickel_ar2021_operations'],
                    source_table='Norilsk Nickel Group saleable metals production / Harjavalta',
                    observation_type='operator_reported_origin_linked_output',
                    estimand='facility_output_metal_content_not_national_export_value',
                    mine_origin_caveat='own_Russian_feed_is_operator_provenance_not_independently_audited_mine_allocation'))
        return pd.DataFrame(output)
    raise RuntimeError('Expected 2012-2021 Harjavalta table not found; inspect source.')


def model_diagnostic(bench):
    member = 'figure_source_tables/longitudinal_2012_2024/release_v3/longitudinal_mine_origin_flows_2012_2024.csv.gz'
    with zipfile.ZipFile(PACKAGE / 'Reproduction_Code_and_Derived_Inputs.zip') as archive:
        raw = archive.read(member)
    kept = []
    for part in pd.read_csv(io.BytesIO(raw), compression='gzip', chunksize=300000):
        select = part[(part.exporter_iso == 'FIN') & (part.metal == 'Nickel')]
        kept.append(select.groupby(['year', 'stage', 'inferred_mine_origin_iso'], as_index=False).attributed_value_usd.sum())
    agg = pd.concat(kept).groupby(['year', 'stage', 'inferred_mine_origin_iso'], as_index=False).attributed_value_usd.sum()
    agg['origin_value_share'] = agg.attributed_value_usd / agg.groupby(['year', 'stage']).attributed_value_usd.transform('sum')
    agg.to_csv(OUT / 'model_finland_nickel_export_origin_diagnostic.csv', index=False)
    rus = agg[agg.inferred_mine_origin_iso.eq('RUS')][['year', 'stage', 'origin_value_share']]
    comparison = rus.merge(bench[bench.element.eq('Nickel')][['year', 'own_russian_feed_output_share',
                    'russian_feed_output_share_lower_rounding_aware']], on='year', validate='many_to_one')
    comparison['comparison_status'] = 'different_estimands_no_validation_score'
    comparison.to_csv(OUT / 'facility_national_estimand_diagnostic.csv', index=False)
    return dict(model_source_member=member, compressed_member_sha256=hashlib.sha256(raw).hexdigest(),
                compared_years=sorted(map(int, comparison.year.unique())),
                diagnostic_only=True, model_changed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.download:
        with ThreadPoolExecutor(max_workers=3) as pool:
            manifest = list(pool.map(download, SOURCES.items()))
        (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    bench = extract_table((OUT / 'nornickel_ar2021_operations.html').read_bytes())
    bench.to_csv(OUT / 'harjavalta_origin_linked_output_2012_2021.csv', index=False)
    diagnostic = model_diagnostic(bench)
    ni = bench[bench.element.eq('Nickel')]
    diagnostic.update(rows=len(bench), independent_facilities=1, years=10,
        elements=['Nickel', 'Copper', 'Palladium', 'Platinum'],
        nickel_share_2017_2021_min=float(ni[ni.year.ge(2017)].own_russian_feed_output_share.min()),
        nickel_share_2017_2021_max=float(ni[ni.year.ge(2017)].own_russian_feed_output_share.max()),
        validation_correlations_computed=False)
    (OUT / 'extraction_checks.json').write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
    print(ni[['year', 'total_saleable_output', 'own_russian_feed_output', 'own_russian_feed_output_share']].to_string(index=False))
    print(json.dumps(diagnostic, indent=2))


if __name__ == '__main__':
    main()
