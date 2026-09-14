"""Independent LP verification, contract tests, hashes and source invariants."""
from dataclasses import replace
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_process/runtime_support/scipy_scope_vendor'))
import numpy as np
import pandas as pd
import scipy
import pypdf
from scipy.optimize import linprog
from provenance_ledger import Ledger,Scope,product_bound,export_lower
from build_constraints import OUT,EXT,source_checks,sha

def solve_product(row):
    # Variables: total, own feed, product nickel, own feed in product.
    a=np.array([[-row.product_share_high,0,1,0],
        [row.product_share_low,0,-1,0],[0,-1,0,1],
        [0,0,-1,1],[-1,1,1,-1]])
    bounds=[(row.total_low,row.total_high),(row.own_low,row.own_high),(0,None),(0,None)]
    result=linprog([0,0,0,1],A_ub=a,b_ub=np.zeros(5),bounds=bounds,method='highs')
    assert result.success
    assert abs(result.fun-row.own_feed_ni_lower)<1e-7
    low,high=0.,1.
    for _ in range(45):
        q=(low+high)/2
        ratio=linprog([0,0,-q,1],A_ub=a,b_ub=np.zeros(5),bounds=bounds,method='highs')
        assert ratio.success
        if ratio.fun>=0:
            low=q
        else:
            high=q
    ratio_error=abs((low+high)/2-row.own_feed_share_lower)
    assert ratio_error<1e-9
    return dict(product=row.product,quantity_lp_error=abs(result.fun-row.own_feed_ni_lower),fraction_lp_error=ratio_error)

def main():
    source_checks()
    products=pd.read_csv(OUT/'facility_product_origin_bounds_2018.csv')
    lpchecks=[solve_product(r) for r in products.itertuples()]
    mass=pd.read_csv(OUT/'national_export_mass_bounds_2018.csv')
    inputs=json.loads((OUT/'national_bridge_inputs.json').read_text())
    errors=[]
    for r in mass.itertuples():
        if np.isinf(r.additional_allowance_t):
            assert r.russian_feed_export_mass_share_lower==0
            continue
        own=inputs['known_own_Russian_feed_output_lower_t_Ni']
        unrelated=r.unassigned_supply_upper_t+r.additional_allowance_t
        # Exported and non-exported amounts of the two source groups.
        opt=linprog([1,0,0,0],A_eq=[[1,1,0,0],[1,0,1,0],[0,1,0,1]],
            b_eq=[r.export_ni_lower_t,own,unrelated],bounds=[(0,None)]*4,method='highs')
        assert opt.success
        error=abs(opt.fun/r.export_ni_lower_t-r.russian_feed_export_mass_share_lower)
        assert error<1e-10
        errors.append(error)
    for _,g in mass.groupby(['target','purity_floor']):
        assert np.all(np.diff(g.sort_values('additional_allowance_t').russian_feed_export_mass_share_lower)<=1e-12)
    for r in products.itertuples():
        widened=product_bound(r.total_low-.5,r.total_high+.5,r.own_low-.5,r.own_high+.5,
            r.product_share_low-.005,r.product_share_high+.005)
        assert widened['own_feed_share_lower']<=r.own_feed_share_lower
    rng=np.random.default_rng(20260908)
    scope=Scope('Harjavalta',2018,'saleable_nickel','tonnes_Ni','operator_own_Russian_feed')
    for _ in range(500):
        known=rng.dirichlet(np.ones(5))[:4]
        ledger=Ledger(scope,1.,dict(zip('ABCD',known)))
        prior=dict(zip('ABCDE',rng.dirichlet(np.ones(5))))
        completed=ledger.complete_residual(prior)
        assert abs(sum(completed.values())-1)<1e-12
        assert all(completed[k]>=v-1e-12 for k,v in ledger.known_lower.items())
        hhi=sum(x*x for x in completed.values())
        lo,hi=ledger.hhi_outer_bounds()
        assert lo-1e-12<=hhi<=hi+1e-12
    unknown=Ledger(scope,1.,{})
    assert unknown.hhi_outer_bounds()==(0.,1.)
    assert unknown.share_interval('RUS')==(0.,1.)
    full=Ledger(scope,1.,{'RUS':1.})
    assert full.complete_residual({'FIN':1.})=={'RUS':1.,'FIN':0.}
    mismatches=[replace(scope,entity='FIN_national_exports'),replace(scope,year=2024),
        replace(scope,product='HS750210'),replace(scope,unit='USD'),replace(scope,provenance='audited_mine_origin')]
    for request in mismatches:
        try:
            full.require_scope(request)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched evidence was accepted.')
    for total,known in [(0,{}),(1,{'RUS':1.1}),(1,{'RUS':-.1}),(1,{'UNKNOWN':1.})]:
        try:
            Ledger(scope,total,known)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid ledger accepted.')
    facility=pd.read_csv(OUT/'facility_partial_provenance_2012_2021.csv')
    assert list(facility.year)==list(range(2012,2022))
    assert (facility.loc[facility.year.le(2014),'russian_feed_share_upper']==1).all()
    assert facility.loc[facility.year.le(2014),'russian_feed_share_lower'].eq(0).all()
    records=pd.read_csv(OUT/'finland_hs750210_2018_record_audit.csv')
    excluded=records[~records.retained_in_previous_model]
    assert len(excluded)==1 and excluded.importer_iso.iloc[0]=='S19'
    reconciliation=pd.read_csv(OUT/'national_facility_output_reconciliation.csv')
    assert reconciliation.loc[reconciliation.discrepancy_exceeds_rounding,'year'].tolist()==[2017]
    assert reconciliation.loc[reconciliation.year.eq(2018),'national_minus_facility_t'].iloc[0]==0
    admissions=pd.read_csv(OUT/'constraint_admission_register.csv')
    assert admissions.status.eq('not_admitted').sum()==3
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    for path,digest in manifest['source_hashes'].items():
        assert sha(ROOT/path)==digest
    package=ROOT/'manuscript_package_20260907_final'
    inventory=pd.read_csv(package/'PACKAGE_SHA256.csv')
    for r in inventory.itertuples():
        assert sha(package/r.file)==r.sha256
    archive_hash=sha(ROOT/'manuscript_package_20260907_final.zip')
    assert archive_hash=='0226e5422850f2b2adab605b47fc8e35d7b41d19a6899ba647e5696d96b84019'
    report=dict(status='PASS',runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,pypdf=pypdf.__version__),
        code_hashes={p.name:sha(p) for p in HERE.glob('*.py')},
        product_lp_checks=lpchecks,national_mass_lp_checks=len(errors),
        maximum_mass_share_lp_error=max(errors),random_conservation_and_known_preservation_tests=500,
        scope_mismatch_rejections=len(mismatches),all_unknown_is_interval_not_zero=True,
        missing_years_not_imputed=True,source_vintage_discrepancy_retained=True,
        independent_facilities=1,unchanged_package_files=len(inventory),archive_sha256=archive_hash,
        adopted_national_value_origin_constraints=0,interpretation='Numerical verification, not independent empirical validation.')
    (OUT/'independent_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='OUTPUT_SHA256.csv']
    pd.DataFrame([dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]).to_csv(OUT/'OUTPUT_SHA256.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
