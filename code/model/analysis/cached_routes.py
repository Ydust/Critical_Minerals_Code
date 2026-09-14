"""Cached equivalent route aggregation for already unique, verified route keys."""
import pandas as pd
import numpy as np

def install(lm,path,edges,direct):
    original=lm.route_stats
    table=pd.read_csv(path).fillna({'chokepoints':''}).rename(columns={'origin':'exporter_iso','dest':'importer_iso'})
    key=['metal','exporter_iso','importer_iso']
    assert not table.duplicated(key).any()
    table=table[key+['mode','chokepoints']]
    def cached(flows,stats,requested_path):
        assert str(requested_path)==str(path)
        lanes=table[table.metal.isin(flows.metal.unique())]
        routed=flows.merge(lanes,on=key,how='left',indicator=True,validate='many_to_one')
        routed['route_matched_value_usd']=np.where(routed['_merge'].eq('both'),routed.reconstructed_value_usd,0.)
        routed['sea_value_usd']=np.where(routed['mode'].eq('sea'),routed.reconstructed_value_usd,0.)
        cover=routed.groupby(lm.GROUP,as_index=False).agg(route_matched_value_usd=('route_matched_value_usd','sum'),sea_value_usd=('sea_value_usd','sum'))
        cover=cover.merge(stats[lm.GROUP+['direct_total_value_usd']],on=lm.GROUP,validate='one_to_one')
        cover['route_coverage_share']=cover.route_matched_value_usd/cover.direct_total_value_usd
        cover['sea_value_share']=cover.sea_value_usd/cover.direct_total_value_usd
        sea=routed[routed['mode'].eq('sea')&routed.chokepoints.fillna('').ne('')].copy()
        if len(sea):
            sea['chokepoint']=sea.chokepoints.str.split('|')
            sea=sea.explode('chokepoint')
            amounts=sea.groupby(lm.GROUP+['chokepoint'],as_index=False).reconstructed_value_usd.sum()
            c=amounts.groupby(lm.GROUP,as_index=False).agg(chokepoint_count=('chokepoint','nunique'),top_chokepoint_value_usd=('reconstructed_value_usd','max'))
            top=lm.top_label(amounts,lm.GROUP,'reconstructed_value_usd','chokepoint','top_chokepoint')
            cover=cover.merge(c.merge(top,on=lm.GROUP),on=lm.GROUP,how='left',validate='one_to_one')
        else:
            cover['chokepoint_count']=0; cover['top_chokepoint_value_usd']=0.; cover['top_chokepoint']=''
        cover['chokepoint_count']=cover.chokepoint_count.fillna(0).astype(int)
        cover['top_chokepoint_value_usd']=cover.top_chokepoint_value_usd.fillna(0.)
        cover['top_chokepoint']=cover.top_chokepoint.fillna('')
        cover['top_chokepoint_share']=cover.top_chokepoint_value_usd/cover.direct_total_value_usd
        return cover.drop(columns='direct_total_value_usd')
    expected=original(edges,direct,path).set_index(lm.GROUP).sort_index()
    actual=cached(edges,direct,path).set_index(lm.GROUP).sort_index()
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_exact=False,rtol=1e-12,atol=1e-12)
    lm.route_stats=cached
    return dict(validated_rows=len(actual),checked_columns=list(expected.columns),relative_tolerance=1e-12)
