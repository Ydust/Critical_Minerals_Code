"""Corrected Fig. 4 reference tables and endpoint cohort, without package mutation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results'
HIST = ROOT / 'historical_expanded_comtrade'
OUT = ROOT / 'structural_integration'
KEY = ['importer_iso', 'metal', 'stage']


def main():
    OUT.mkdir(exist_ok=True)
    annual = pd.read_csv(HIST / 'annual_indicators.csv.gz')
    windows = pd.read_csv(HIST / 'all_windows.csv.gz')
    # Reapply the published endpoint rule; do not impose the historical 704 count.
    eligible = annual[(annual.direct_total_value_usd >= 1e6)
                      & (annual.origin_coverage_ratio >= .8)
                      & (annual.route_coverage_share >= .5)]
    baseline = eligible[eligible.year.eq(2020)][KEY]
    endpoint = eligible[eligible.year.eq(2024) & eligible.direct_total_value_usd.ge(1e7)]
    cohort = endpoint.merge(baseline, on=KEY, validate='one_to_one')
    cohort.to_csv(OUT / 'fixed_endpoint_cohort.csv', index=False)
    active = annual.merge(cohort[KEY], on=KEY, validate='many_to_one')
    active[active.direct_total_value_usd.gt(0)].groupby('year').size().rename('active_units').to_csv(OUT / 'fixed_cohort_annual_counts.csv')
    primary = windows[windows.window_family.eq('adjacent')
                      & windows.three_layer_evidence_eligible
                      & np.isclose(windows.classification_threshold, .025)].copy()
    assert not primary.duplicated(KEY + ['year']).any()
    events = primary[primary.classification.isin(['route_transfer', 'multiple_risk_transfer'])].copy()
    events = events.sort_values(['year'] + KEY).reset_index(drop=True)
    events['event_id'] = np.arange(len(events))
    events['transfer_shift_usd'] = events.transition_value_usd * events.delta_top_chokepoint_share
    events['retained'] = events.top_chokepoint.eq(events.top_chokepoint_previous.fillna('None'))
    assert events.delta_top_chokepoint_share.ge(.025 - 1e-12).all()
    trade = pd.read_csv(HIST / 'analytical_trade_panel.csv.gz')
    flows = trade.groupby(KEY + ['year', 'exporter_iso'], as_index=False).reconstructed_value_usd.sum()
    lanes = pd.read_csv(ROOT / 'routes.csv').rename(columns={'origin': 'exporter_iso', 'dest': 'importer_iso'})
    assert not lanes.duplicated(['metal', 'exporter_iso', 'importer_iso']).any()
    routed = events[KEY + ['year', 'event_id', 'top_chokepoint', 'transfer_shift_usd']].merge(flows, on=KEY + ['year'], validate='one_to_many')
    routed = routed.merge(lanes[['metal', 'exporter_iso', 'importer_iso', 'mode', 'chokepoints']], on=['metal', 'exporter_iso', 'importer_iso'], how='left', validate='many_to_one')
    routed = routed[routed['mode'].eq('sea') & routed.chokepoints.notna()].copy()
    routed = routed[[r.top_chokepoint in str(r.chokepoints).split('|') for r in routed.itertuples()]].copy()
    routed['weighted_shift_usd'] = routed.transfer_shift_usd * routed.reconstructed_value_usd / routed.groupby('event_id').reconstructed_value_usd.transform('sum')
    allocated = routed.groupby('event_id').weighted_shift_usd.sum().reindex(events.event_id, fill_value=0).to_numpy()
    np.testing.assert_allclose(allocated, events.transfer_shift_usd, rtol=1e-10, atol=1e-5)
    summaries = []
    paths_all = []
    for year, group in events.groupby('year'):
        r = routed[routed.year.eq(year)]
        paths = r.groupby(['exporter_iso', 'importer_iso', 'top_chokepoint'], as_index=False).agg(weighted_shift_usd=('weighted_shift_usd', 'sum'), event_count=('event_id', 'nunique'))
        paths = paths.sort_values(['weighted_shift_usd', 'exporter_iso', 'importer_iso', 'top_chokepoint'], ascending=[False, True, True, True])
        paths['year'] = year
        paths['rank'] = np.arange(1, len(paths) + 1)
        paths['cumulative_shift_share'] = paths.weighted_shift_usd.cumsum() / paths.weighted_shift_usd.sum()
        paths_all.append(paths)
        chokes = group.groupby('top_chokepoint').transfer_shift_usd.sum().sort_values(ascending=False)
        apparent = primary[primary.year.eq(year) & primary.apparent_derisking]
        summaries.append(dict(year=int(year), event_count=len(group), apparent_count=len(apparent),
                              event_value_usd=float(group.transition_value_usd.sum()),
                              apparent_value_usd=float(apparent.transition_value_usd.sum()),
                              transfer_shift_usd=float(group.transfer_shift_usd.sum()),
                              dominant_chokepoint=chokes.index[0],
                              top32_shift_share=float(paths.head(32).weighted_shift_usd.sum() / paths.weighted_shift_usd.sum()),
                              retained_value_share=float(group.loc[group.retained, 'transition_value_usd'].sum() / group.transition_value_usd.sum())))
    paths = pd.concat(paths_all, ignore_index=True)
    mineral = events.groupby(['year', 'metal'], as_index=False).agg(event_count=('event_id', 'size'), endpoint_value_usd=('transition_value_usd', 'sum'), incremental_shift_usd=('transfer_shift_usd', 'sum'))
    mineral['incremental_shift_share'] = mineral.incremental_shift_usd / mineral.groupby('year').incremental_shift_usd.transform('sum')
    np.testing.assert_allclose(mineral.groupby('year').incremental_shift_share.sum(), 1., atol=1e-12)
    events.to_csv(OUT / 'fig4_events.csv.gz', index=False)
    routed.to_csv(OUT / 'fig4_event_route_allocations.csv.gz', index=False)
    paths.to_csv(OUT / 'fig4_all_ranked_paths.csv', index=False)
    paths[paths.year.ge(2021) & paths['rank'].le(32)].to_csv(OUT / 'fig4_displayed_paths.csv', index=False)
    mineral.to_csv(OUT / 'fig4_mineral_composition.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'fig4_annual_summary.csv', index=False)
    retention = {}
    for name, subset in [('all_12', events), ('exclude_edition_boundaries', events[~events.year.isin([2015, 2020])])]:
        retention[name] = dict(event_count=len(subset), retained_value_share=float(subset.loc[subset.retained, 'transition_value_usd'].sum() / subset.transition_value_usd.sum()))
    inputs = [HIST / 'annual_indicators.csv.gz', HIST / 'all_windows.csv.gz', HIST / 'analytical_trade_panel.csv.gz', ROOT / 'routes.csv', Path(__file__)]
    report = dict(fixed_cohort_units=len(cohort), recent_route_events=int(events.year.ge(2021).sum()),
                  all_route_events=len(events), retention=retention,
                  max_relative_event_allocation_error=float(np.max(np.abs(allocated - events.transfer_shift_usd.to_numpy()) / events.transfer_shift_usd.to_numpy())),
                  recent_windows=[r for r in summaries if r['year'] >= 2021],
                  input_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  scope='Corrected reference only. Static route exposure, conditional origin attribution. No structural sensitivity intervals or submission readiness claim.',
                  still_pending=['Fig2 mismatch and country roles', 'Fig3 event networks', 'Fig4 route-ranking sensitivity', 'Fig5 objective ablations', 'figure rendering and package synchronization'])
    (OUT / 'STRUCTURAL_REFERENCE_QA.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
