"""Independent checks of extracted facts, conditional bounds and draw outputs."""
from pathlib import Path
import hashlib
import json
import unittest
from decimal import Decimal
import numpy as np
import pandas as pd
from lxml import html
from run_historical_joint_sensitivity import classify, draw_inputs, ar_errors

ROOT=Path(__file__).resolve().parent


class EvidenceTests(unittest.TestCase):
    def test_raw_source_hashes(self):
        folder=ROOT/'external/facility_supply'
        records=json.loads((folder/'source_manifest.json').read_text())
        self.assertEqual(len(records),6)
        for record in records:
            self.assertEqual(hashlib.sha256((folder/record['file']).read_bytes()).hexdigest(),record['sha256'])

    def test_independent_html_table_extraction(self):
        # pandas HTML-table parser is independent of the lxml row-walking extractor.
        folder=ROOT/'external/facility_supply'
        tables=pd.read_html(folder/'nornickel_ar2021_operations.html')
        table=next(t for t in tables if len(t.columns)==11 and '2012' in str(t.columns))
        position=next(i for i,x in enumerate(table.iloc[:,0].astype(str)) if 'HARJAVALTA (FINLAND)' in x)
        bench=pd.read_csv(folder/'harjavalta_origin_linked_output_2012_2021.csv')
        self.assertEqual(len(bench),40)
        self.assertFalse(bench.duplicated(['element','year']).any())
        for j,element in enumerate(['Nickel','Copper','Palladium','Platinum']):
            for column,year in enumerate(range(2012,2022),1):
                row=bench[(bench.element==element)&(bench.year==year)].iloc[0]
                total=int(table.iloc[position+1+2*j,column])
                own=int(table.iloc[position+2+2*j,column])
                self.assertEqual(total,row.total_saleable_output)
                self.assertEqual(own,row.own_russian_feed_output)
                self.assertAlmostEqual(float(Decimal(own)/Decimal(total)),row.own_russian_feed_output_share,places=14)
                self.assertLessEqual(row.russian_feed_output_share_lower_rounding_aware,row.own_russian_feed_output_share+1e-14)

    def test_route_bounds_contain_grid(self):
        # Enumerate one-chokepoint missing incidence at each endpoint. This is a
        # subset of the allowed multi-chokepoint configurations, not a sharpness proof.
        rng=np.random.default_rng(997)
        for _ in range(40):
            r0,r1=rng.uniform(.5,1,2)
            c0,c1=rng.uniform(0,r0),rng.uniform(0,r1)
            matrix=np.array([[1e7,.7,.55,.4,r0,c0],[2e7,.4,.3,.4,r1,c1]])
            result=classify(matrix,(np.array([0]),np.array([1])))
            for missing0 in np.linspace(0,1-r0,9):
                for missing1 in np.linspace(0,1-r1,9):
                    change=(c1+missing1)-(c0+missing0)
                    transfer=change>=.025
                    substantive=change<=-.025
                    self.assertLessEqual(bool(result['transfer_lower'][0]),transfer)
                    self.assertLessEqual(transfer,bool(result['transfer_upper'][0]))
                    self.assertLessEqual(bool(result['substantive_lower'][0]),substantive)
                    self.assertLessEqual(substantive,bool(result['substantive_upper'][0]))

    def test_draw_reproducibility_and_pairing(self):
        sizes=dict(supplier=4,lane=7,producer=3)
        a=draw_inputs(sizes,3,0.)
        b=draw_inputs(sizes,3,0.)
        c=draw_inputs(sizes,3,.8)
        for key in ['supplier','lane','producer']:
            np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(a[key][:,0],c[key][:,0])
        self.assertEqual(a['weights'],c['weights'])
        self.assertTrue(all(0<=x<=1 for x in a['weights'].values()))

    def test_matte_source_partition_and_coverage(self):
        folder=ROOT/'external/nickel_matte_coverage'
        manifest=json.loads((folder/'source_manifest.json').read_text())
        self.assertEqual({r['year'] for r in manifest},set(range(2012,2025)))
        summary=pd.read_csv(folder/'nickel_matte_omission_summary.csv').set_index('year')
        for record in manifest:
            raw=(folder/record['file']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),record['sha256'])
            records=json.loads(raw)['data']
            self.assertLess(len(records),500)
            world=next(r for r in records if r['partnerCode']==0)
            value=Decimal(str(world['primaryValue']))
            total=sum(Decimal(str(r['primaryValue'])) for r in records if r['partnerCode']!=0)
            self.assertLess(abs(total-value)/value,Decimal('1e-6'))
            self.assertAlmostEqual(float(value),summary.loc[record['year'],'reported_matte_world_import_usd'],places=5)
        self.assertAlmostEqual(summary.loc[2018,'reported_matte_world_import_usd'],917717873.363,places=5)
        # Independent public WITS display rounds this same source to USD 1000.
        self.assertLess(abs(summary.loc[2018,'reported_matte_world_import_usd']-917717.87*1000),5.)
        self.assertGreater(summary.loc[2018,'omitted_matte_to_selected_nickel_value_ratio'],4.7)

    def test_all_500_draws_and_membership(self):
        folder=ROOT/'results/historical_joint_sensitivity'
        manifest=json.loads((folder/'run_manifest.json').read_text())
        self.assertEqual(manifest['actual_draws'],500)
        self.assertEqual(manifest['windows'],111480)
        self.assertEqual(manifest['unit_years'],68164)
        self.assertLess(max(manifest['baseline_relative_errors'].values()),1e-10)
        data=pd.read_csv(folder/'window_summary_all_draws.csv')
        for rho in [0.,.8]:
            d=data[(data.temporal_rho==rho)&(data.window_family=='adjacent')&(data.end_year=='pooled')&(data.metal=='all')]
            self.assertEqual(len(d),250)
            self.assertEqual(set(d.draw),set(range(1,251)))
            for prefix in ['transfer','substantive']:
                self.assertTrue((d[prefix+'_lower_share']<=d[prefix+'_share']+1e-12).all())
                self.assertTrue((d[prefix+'_share']<=d[prefix+'_upper_share']+1e-12).all())
            membership=pd.read_csv(folder/f'classification_membership_rho_{rho:.1f}.csv.gz')
            self.assertEqual(len(membership),111480)
            for key in ['eligible','apparent','transfer','substantive']:
                self.assertTrue(membership[key+'_draw_count'].between(0,250).all())
            self.assertTrue((membership.transfer_draw_count+membership.substantive_draw_count<=membership.apparent_draw_count).all())
        self.assertFalse(manifest['coverage']['allocation_resolved_each_draw'])
        self.assertFalse(manifest['coverage']['raw_mirror_reconstruction'])

    def test_baci_extraction_against_preexisting_selected_file(self):
        folder=ROOT/'external/baci_nickel_intermediates'
        raw=pd.read_csv(folder/'baci_nickel_six_products_2007_2024_raw.csv.gz')
        old=pd.read_csv(ROOT.parents[1]/'external_sources/baci_hs07_202601/baci_hs07_202601_fig5_selected_2007_2024.csv.gz')
        keys=['t','i','j','k']
        columns=keys+['v','q']
        codes=[260400,282540,750210]
        a=raw[raw.k.isin(codes)][columns].sort_values(keys).reset_index(drop=True)
        b=old[old.k.isin(codes)][columns].sort_values(keys).reset_index(drop=True)
        self.assertEqual(len(a),29103)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-12,atol=1e-12)
        prepared=pd.read_csv(folder/'baci_nickel_six_products_2007_2024_prepared.csv.gz')
        self.assertEqual(len(prepared),47702)
        np.testing.assert_allclose(prepared.trade_value_usd,prepared.value_thousand_usd*1000,rtol=1e-12)
        np.testing.assert_allclose(prepared.reported_product_mass_kg,prepared.quantity_tonnes*1000,rtol=1e-12,equal_nan=True)
        self.assertEqual(prepared.reported_product_mass_kg.isna().sum(),raw.q.isna().sum())


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    checks=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
    (ROOT/'results/followup_independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
