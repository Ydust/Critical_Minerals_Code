"""Explain prior attenuation and selection changes without tuning the model."""
from __future__ import annotations

import json
import zipfile

from run_scope_reanalysis import ARCHIVE, CONFIGS, IP, OUT, GROUP, KEYS
from validate_and_compare import independent_profiles
import numpy as np
import pandas as pd


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        seed = pd.read_csv(z.open(IP+'mine_origin_seed_bgs_wmd_2008_2024.csv.gz'), compression='gzip')
    seed = seed[seed.metal.eq('Nickel')]
    records = []
    for year, group in seed.groupby('year'):
        share = group.loc[group.mine_origin_iso.eq('RUS'), 'production'].sum()/group.production.sum()
        records.append(dict(year=int(year), russian_global_production_prior=float(share), metal_local_weight=.45,
            russian_export_share_upper_bound_given_metal_prior_blend=float(.45+.55*share)))
    pd.DataFrame(records).to_csv(OUT/'russian_origin_prior_blend_upper_bound.csv', index=False)

    eligibility, coverage = [], []
    for name in CONFIGS:
        annual = pd.read_csv(OUT/name/'annual_indicators.csv.gz')
        early = annual.rename(columns={c: c+'_start' for c in annual if c not in KEYS})
        early['year'] = early.year_start+4
        pairs = annual[annual.year.ge(2012)].merge(early, on=KEYS+['year'], how='left', validate='one_to_one')
        pairs['meets_current_value'] = pairs.direct_total_value_usd.ge(1e7)
        pairs['meets_earlier_value'] = pairs.direct_total_value_usd_start.ge(1e6)
        pairs['meets_current_route'] = pairs.route_coverage_share.ge(.5)
        pairs['meets_earlier_route'] = pairs.route_coverage_share_start.ge(.5)
        pairs['meets_current_origin'] = pairs.origin_coverage_ratio.ge(.8)
        pairs['meets_earlier_origin'] = pairs.origin_coverage_ratio_start.ge(.8)
        gates = [c for c in pairs if c.startswith('meets_')]
        pairs['eligible'] = pairs[gates].all(axis=1)
        pairs.insert(0, 'variant', name)
        pairs['failed_gates'] = pairs[gates].apply(lambda row: '|'.join(k for k, v in row.items() if not v), axis=1)
        eligibility.append(pairs[['variant', 'year', 'importer_iso', 'stage', 'direct_total_value_usd', 'direct_total_value_usd_start', 'route_coverage_share', 'route_coverage_share_start', 'eligible', 'failed_gates']])
        for year, group in pairs.groupby('year'):
            for stage in ['all']+sorted(group.stage.unique()):
                sub = group if stage == 'all' else group[group.stage.eq(stage)]
                total = sub.direct_total_value_usd.sum()
                coverage.append(dict(variant=name, year=int(year), stage=stage, all_unit_value_usd=float(total),
                    selected_unit_value_usd=float(sub.loc[sub.eligible, 'direct_total_value_usd'].sum()),
                    selected_value_fraction=float(sub.loc[sub.eligible, 'direct_total_value_usd'].sum()/total),
                    all_units=len(sub), selected_units=int(sub.eligible.sum())))
    all_eligibility = pd.concat(eligibility, ignore_index=True)
    all_eligibility.to_csv(OUT/'allocation_eligibility_diagnostic.csv.gz', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'allocation_selected_coverage.csv', index=False)
    # Identify the specific new corridors responsible for the 2020 screen.
    lanes = pd.read_csv(OUT.parent/'frozen/figure_source_tables/longitudinal_2012_2024/input_snapshot/minerals_lanes.csv')
    nickel = lanes[lanes.metal.eq('Nickel')]
    known = set(zip(nickel.origin, nickel.dest))
    trade = pd.read_csv(OUT/'expanded6/trade_corridors.csv.gz')
    sub = trade[trade.year.eq(2020) & trade.stage.eq('material') & trade.importer_iso.isin(['CHN', 'CAN', 'NLD'])].copy()
    sub['route_template_available'] = [(e, i) in known for e, i in zip(sub.exporter_iso, sub.importer_iso)]
    sub['importer_value_share'] = sub.reconstructed_value_usd/sub.groupby('importer_iso').reconstructed_value_usd.transform('sum')
    sub.sort_values(['importer_iso', 'reconstructed_value_usd'], ascending=[True, False]).to_csv(OUT/'material_2020_route_screen_corridors.csv', index=False)
    print(pd.DataFrame(coverage).query('year == 2024 and stage == "all"').to_string(index=False))
    print(sub[~sub.route_template_available].sort_values('reconstructed_value_usd', ascending=False).head(10).to_string(index=False))


if __name__ == '__main__':
    main()
