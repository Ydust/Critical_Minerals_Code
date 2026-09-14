"""Extract financial source facts; verify a scoped sales bound and stock gap."""
from pathlib import Path
import hashlib,json,re,sys
from lxml import html
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
EXT=HERE/'external'
sys.path.insert(0,str(ROOT/'research_process/runtime_support/scipy_scope_vendor'))
from scipy.optimize import linprog
import scipy
sys.path.insert(0,str(ROOT/'research_process/origin_constraints_20260908'))
from provenance_ledger import Scope,Ledger,export_lower

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def clean(node):return ' '.join(node.text_content().split())
def rows(table):return [[clean(c) for c in tr.xpath('./th|./td')] for tr in table.xpath('.//tr')]

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 sources=json.loads((EXT/'source_manifest.json').read_text())
 for r in sources:assert sha(EXT/r['file'])==r['sha256']
 rev=html.fromstring((EXT/'revenue_2018.html').read_bytes())
 rt=clean(rev)
 assert 'USD 13,531 per tonne' in rt and 'to 208 thousand tons' in rt
 assert 'Sales volume of refined nickel produced from own Russian feed' in rt
 assert 'produced from third-party feed decreased' in rt
 stat=html.fromstring((EXT/'statements_2018.html').read_bytes())
 st=clean(stat)
 assert 'US Dollars million' in st
 assert 'lower of net cost of production or net realisable value' in st
 assert 'external customers breakdown by metal' in st
 tables=stat.xpath('//table')
 seg=next(t for t in tables if 'NN Harjavalta' in clean(t) and 'Nickel' in clean(t) and 'For the year ended 31 December 2018' in clean(t))
 sr=rows(seg)
 nr=next(r for r in sr if r[0]=='Nickel')
 assert nr==['Nickel','1,827','275','805','53','53','3,013'],nr
 pd.DataFrame([dict(segment=k,nickel_external_revenue_usd_m=float(v.replace(',','')),year=2018,
     source_id='statements_2018',source_section='Note 6 external metal sales by segment and metal')
     for k,v in zip(['GMK Group','KGMK Group','NN Harjavalta','Other mining','Other non-metallurgical','Total'],nr[1:])]).to_csv(OUT/'segment_nickel_external_sales_2018.csv',index=False)
 inv=next(t for t in tables if 'Refined metals and other metal products' in clean(t))
 ir=rows(inv)
 inventory=[]
 for r in ir:
  if len(r)==3 and r[0] and r[0]!='At 31 December 2018':
   try:a,b=[float(x.replace(',','').replace('(','-').replace(')','')) for x in r[1:]]
   except ValueError:continue
   for year,val in [(2018,a),(2017,b)]:
    inventory.append(dict(year=year,item=r[0],value=val,unit='USD_million_book_value',entity='Nornickel_consolidated_group',product='multi_metal_or_supplies',admissible_as_Finland_nickel_tonnes=False))
 pd.DataFrame(inventory).to_csv(OUT/'inventory_book_values_not_nickel_mass.csv',index=False)
 assert next(r['value'] for r in inventory if r['year']==2017 and r['item']=='Refined metals and other metal products')==655
 facts=dict(year=2018,own_Russian_feed_refined_nickel_sales_t=208000,own_feed_average_realized_usd_per_t=13531,
     group_nickel_external_revenue_usd_m=3013,harjavalta_nickel_external_revenue_usd_m=805,
     quantity_price_scope_alignment='same category in MD&A nickel section; conditional on stated reporting scopes',
     provenance='operator_own_Russian_feed_not_independently_audited_mine_origin')
 (OUT/'sales_source_facts.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
 results=[]
 for scenario,mult in [('nearest_displayed_unit',1),('doubled_rounding_envelope',2)]:
  rlo=(208000-500*mult)*(13531-.5*mult)/1e6
  rhi=(208000+500*mult)*(13531+.5*mult)/1e6
  glo,ghi=3013-.5*mult,3013+.5*mult
  flo,fhi=805-.5*mult,805+.5*mult
  residual=ghi-rlo
  amount=max(0.,flo-residual)
  fraction=max(0.,1-residual/flo)
  # Variables G, F, R, own-feed Harjavalta revenue x, all USD million.
  a=[[0,1,0,0],[-1,0,1,0],[0,-1,0,1],[0,0,-1,1],[-1,1,1,-1]]
  b=[ghi,0,0,0,0]
  bounds=[(glo,ghi),(flo,fhi),(rlo,rhi),(0,None)]
  q=linprog([0,0,0,1],A_ub=a,b_ub=b,bounds=bounds,method='highs')
  assert q.success and abs(q.fun-amount)<1e-9
  lo,hi=0.,1.
  for _ in range(45):
   mid=(lo+hi)/2
   f=linprog([0,-mid,0,1],A_ub=a,b_ub=b,bounds=bounds,method='highs')
   assert f.success
   if f.fun>=0:lo=mid
   else:hi=mid
  error=abs((lo+hi)/2-fraction)
  assert error<1e-10
  results.append(dict(scenario=scenario,group_own_feed_refined_revenue_lower_usd_m=rlo,
      group_remaining_revenue_upper_usd_m=residual,harjavalta_own_feed_revenue_lower_usd_m=amount,
      harjavalta_own_feed_revenue_share_lower=fraction,harjavalta_Russian_feed_revenue_share_upper=1.,
      fraction_lp_error=error,scope='segment_all_nickel_external_revenue_not_national_customs_exports'))
 pd.DataFrame(results).to_csv(OUT/'segment_origin_revenue_bounds_2018.csv',index=False)
 # An exactly known net change still admits arbitrarily large opening stocks.
 stocktests=[]
 for delta in [-1000.,0.,1000.]:
  r=linprog([-1,0],A_eq=[[-1,1]],b_eq=[delta],bounds=[(0,None),(0,None)],method='highs')
  assert r.status==3
  stocktests.append(dict(synthetic_net_change_t=delta,status='opening_stock_unbounded',observed_stock_data=False))
 pd.DataFrame(stocktests).to_csv(OUT/'net_change_identification_tests.csv',index=False)
 scope=Scope('NN_Harjavalta_segment',2018,'all_nickel_external_revenue','USD','operator_own_Russian_feed')
 ledger=Ledger(scope,1.,{'RUS':results[0]['harjavalta_own_feed_revenue_share_lower']})
 target=Scope('Finland_national_exports',2018,'HS750210','USD','mine_origin')
 rejected=False
 try:ledger.require_scope(target)
 except ValueError:rejected=True
 assert rejected
 prior=ROOT/'research_process/origin_constraints_20260908/results/national_export_mass_bounds_2018.csv'
 national=pd.read_csv(prior)
 assert national.loc[np.isinf(national.additional_allowance_t),'russian_feed_export_mass_share_lower'].eq(0).all()
 admissions=[dict(evidence='Group metal inventories',status='not_admitted_as_nickel_mass_cap',reason='book value, multiple metals, group geography and valuation basis'),
  dict(evidence='Known net inventory change',status='not_admitted_as_opening_stock_cap',reason='stock equation has an unbounded opening/closing balance direction'),
  dict(evidence='Group own-feed refined sales and segment nickel revenue',status='conditional_segment_revenue_lower_bound',reason='same-year source categories, rounded disclosures and nonnegative segment allocation'),
  dict(evidence='Segment revenue source lower bound',status='not_admitted_to_national_export_fingerprint',reason='entity, product and provenance mismatch; domestic sales/customs bridge absent')]
 pd.DataFrame(admissions).to_csv(OUT/'evidence_admission.csv',index=False)
 needed=[('opening_and_closing_stock','2017-12-31 and 2018-12-31; Finland; nickel tonnes; owner and provenance','bounds prior-stock supply'),
  ('other_supply','2018; secondary conversion and supplies not already included in refining output','bounds additional unaccounted supply without double counting'),
  ('segment_customs_bridge','2018; Harjavalta nickel revenue by product and domestic/export destination; customs reconciliation','maps segment nickel revenue to national HS750210 FOB value'),
  ('source_validation','matched material source and mine-location records','distinguishes operator feed labels from audited mine origin')]
 pd.DataFrame(needed,columns=['field','required_scope','purpose']).to_csv(OUT/'minimum_missing_bridge_fields.csv',index=False)
 package=ROOT/'manuscript_package_20260907_final'
 inventoryfiles=pd.read_csv(package/'PACKAGE_SHA256.csv')
 for row in inventoryfiles.itertuples():assert sha(package/row.file)==row.sha256
 archive=sha(ROOT/'manuscript_package_20260907_final.zip')
 assert archive=='0226e5422850f2b2adab605b47fc8e35d7b41d19a6899ba647e5696d96b84019'
 report=dict(status='PASS',source_archives=len(sources),revenue_bound_scenarios=2,stock_unbounded_lp_tests=3,
     national_scope_rejected=True,national_extra_supply_upper_bound_identified=False,
     national_model_modified=False,unchanged_package_files=len(inventoryfiles),archive_sha256=archive,
     prior_mass_bound_file_sha256=sha(prior),runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
     code_sha256=sha(__file__),results=results)
 (OUT/'independent_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 paths=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='OUTPUT_SHA256.csv']
 pd.DataFrame([dict(path=str(p.relative_to(HERE)),sha256=sha(p)) for p in sorted(paths)]).to_csv(OUT/'OUTPUT_SHA256.csv',index=False)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
