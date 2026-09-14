"""Logical and independently aggregated checks of the candidate bounds."""
import csv
import gzip
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from analyse_identification_bounds import bounds

ROOT=Path(__file__).resolve().parent


class BoundTests(unittest.TestCase):
    def test_complete_routes_coincide(self):
        frame=pd.DataFrame(dict(top_chokepoint_share_previous=[.1,.4,.2],top_chokepoint_share=[.2,.3,.2],
                                route_coverage_share_previous=[1.,1.,1.],route_coverage_share=[1.,1.,1.]))
        a,b=bounds(frame,False),bounds(frame,True)
        for x,y in zip(a,b):
            np.testing.assert_allclose(x,y,atol=1e-15)
        self.assertEqual(a[0].tolist(),[True,False,False])
        self.assertEqual(a[1].tolist(),[False,True,False])

    def test_missing_routes_only_widen(self):
        frame=pd.DataFrame(dict(top_chokepoint_share_previous=[.1,.4,.2],top_chokepoint_share=[.2,.3,.2],
                                route_coverage_share_previous=[.5,.8,.9],route_coverage_share=[.6,.8,.9]))
        exact,outer=bounds(frame,False),bounds(frame,True)
        self.assertTrue(np.all(outer[2]<=exact[2]+1e-15))
        self.assertTrue(np.all(outer[3]>=exact[3]-1e-15))
        self.assertTrue(np.all(~outer[0] | exact[0]))
        self.assertTrue(np.all(~exact[1] | outer[1]))

    def test_independent_saved_aggregation(self):
        with gzip.open(ROOT/'results/origin_unrestricted_unit_windows.csv.gz','rt',encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        total=sum(float(x['transition_value_usd']) for x in rows)
        with (ROOT/'results/origin_unrestricted_bounds.csv').open(encoding='utf-8-sig',newline='') as stream:
            summaries={x['route_treatment']:x for x in csv.DictReader(stream) if x['sample']=='pooled'}
        for mode,row in summaries.items():
            lower=sum(float(x['transition_value_usd']) for x in rows if x[mode+'_transfer_sufficient']=='True')/total*100
            upper=sum(float(x['transition_value_usd']) for x in rows if x[mode+'_substantive_necessary']=='True')/total*100
            self.assertAlmostEqual(lower,float(row['transfer_lower_pct']),places=10)
            self.assertAlmostEqual(upper,float(row['substantive_upper_pct']),places=10)
            self.assertAlmostEqual(lower-upper,float(row['transfer_minus_substantive_lower_pp']),places=10)

    def test_portwatch_coverage_and_annual_means(self):
        with gzip.open(ROOT/'external/portwatch/portwatch_daily_2019_2024.csv.gz','rt',encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),17536)
        self.assertEqual(len({(x['portid'],x['date']) for x in rows}),17536)
        sums={}
        for x in rows:
            key=x['portid'],x['year']
            total,count=sums.get(key,(0,0))
            sums[key]=(total+int(x['n_total']),count+1)
        with (ROOT/'external/portwatch/portwatch_annual_2019_2024.csv').open(encoding='utf-8-sig',newline='') as stream:
            for x in csv.DictReader(stream):
                total,count=sums[x['portid'],x['year']]
                self.assertEqual(total,int(x['total_transits']))
                self.assertEqual(count,int(x['expected_days']))
                self.assertAlmostEqual(total/count,float(x['total_transits_daily_mean']),places=10)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BoundTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'results/independent_checks.json').write_text(json.dumps(dict(tests_run=result.testsRun,
          failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
