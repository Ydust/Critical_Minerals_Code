"""Source-grounded facility bounds and a conditional national mass ledger."""
from pathlib import Path
import hashlib
import json
import re
import sys
from lxml import html
import numpy as np
import pandas as pd
from pypdf import PdfReader
from provenance_ledger import Ledger,Scope,product_bound,export_lower

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results'
EXT=HERE/'external'
FAC=ROOT/'research_process/evidence_extension_20260908/external/facility_supply'
BACI=ROOT/'research_process/evidence_extension_20260908/external/baci_nickel_intermediates/baci_nickel_six_products_2007_2024_prepared.csv.gz'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def source_checks():
    manifest=json.loads((EXT/'source_manifest.json').read_text())
    for r in manifest:
        if r['status']=='archived':
            assert sha(EXT/r['file'])==r['sha256']
    old=json.loads((FAC/'source_manifest.json').read_text())
    r=next(x for x in old if x['source_id']=='nornickel_ar2021_operations')
    assert sha(FAC/r['file'])==r['sha256']
    p=PdfReader(EXT/'harjavalta_2018_pdf.pdf')
    text=p.pages[1].extract_text()
    for token in ['Briquettes','Cathodes','50','27','Chemicals and solutions 17','Powders 6','60.77','59.34']:
        assert token in text,token
    official=' '.join(html.fromstring((EXT/'census_ch75_2017.html').read_bytes()).text_content().split())
    assert 'at least 99 percent of nickel plus cobalt' in official and 'does not exceed 1.5 percent' in official
    assert '7502.10.0000' in official
    return manifest

def facility():
    # Re-extract the original source rather than trusting the previous CSV.
    sys.path.insert(0,str(ROOT/'research_process/evidence_extension_20260908'))
    from build_facility_benchmarks import extract_table
    full=extract_table((FAC/'nornickel_ar2021_operations.html').read_bytes())
    saved=pd.read_csv(FAC/'harjavalta_origin_linked_output_2012_2021.csv')
    pd.testing.assert_frame_equal(full,saved,check_dtype=False,rtol=1e-12,atol=1e-12)
    bench=full[full.element.eq('Nickel')].copy()
    rows=[]
    for r in bench.itertuples():
        total=r.total_saleable_output+.5
        lower=max(0,r.own_russian_feed_output-.5)
        ledger=Ledger(Scope('Harjavalta',r.year,'all_saleable_nickel','tonnes_Ni','operator_own_Russian_feed'),total,{'RUS':lower})
        lo,hi=ledger.share_interval('RUS')
        hlo,hhi=ledger.hhi_outer_bounds()
        rows.append(dict(year=r.year,total_reported_t=r.total_saleable_output,known_reported_t=r.own_russian_feed_output,
            denominator_upper_t=total,known_lower_t=lower,unresolved_upper_t=ledger.unresolved,
            russian_feed_share_lower=lo,russian_feed_share_upper=hi,
            country_hhi_outer_lower_if_provenance_equivalent=hlo,country_hhi_outer_upper=hhi,
            source_id=r.source_id,scope='facility_saleable_output_not_national_exports'))
    pd.DataFrame(rows).to_csv(OUT/'facility_partial_provenance_2012_2021.csv',index=False)
    r=bench[bench.year.eq(2018)].iloc[0]
    t,o=float(r.total_saleable_output),float(r.own_russian_feed_output)
    assert t==60765 and o==59337
    assert 60765<=t+.5 and t-.5<=60775 and 59335<=o+.5 and o-.5<=59345
    groups=[('briquettes',.50,.005),('cathodes',.27,.005),('briquettes_and_cathodes',.77,.01),
        ('chemicals_and_solutions',.17,.005),('powders',.06,.005)]
    products=[]
    for name,share,half in groups:
        facts=dict(year=2018,product=name,total_low=t-.5,total_high=t+.5,own_low=o-.5,own_high=o+.5,
            product_share_reported=share,product_share_low=share-half,product_share_high=share+half,
            product_source='harjavalta_2018_pdf_p87',quantity_source='nornickel_ar2021_operations_Harjavalta')
        products.append(facts|product_bound(t-.5,t+.5,o-.5,o+.5,share-half,share+half))
    pd.DataFrame(products).to_csv(OUT/'facility_product_origin_bounds_2018.csv',index=False)
    return bench

def national(bench):
    reader=PdfReader(EXT/'finland_usgs_2017_2018.pdf')
    text=reader.pages[5].extract_text()
    metals=[36639,36350,45606,51342,50435]
    chemicals=[5964,7129,8048,8358,10330]
    for label,expected in [('Metal, electrolytic',metals),('Chemicals',chemicals)]:
        line=next(x for x in text.splitlines() if x.startswith(label))
        numbers=[int(x.replace(',','')) for x in re.findall(r'\d[\d,]*',line)]
        assert numbers==expected,(label,numbers)
    rows=[]
    for year,m,c in zip(range(2014,2019),metals,chemicals):
        facility_total=float(bench.loc[bench.year.eq(year),'total_saleable_output'].iloc[0])
        rows.append(dict(year=year,national_refined_metal_ni_t=m,national_refined_chemical_ni_t=c,
            national_refined_total_ni_t=m+c,facility_total_ni_t=facility_total,
            national_minus_facility_t=m+c-facility_total,
            discrepancy_exceeds_rounding=abs(m+c-facility_total)>1.5,
            source='USGS Table 1 p15.5; Nornickel 2021 retrospective table',
            interpretation='Output-scope cross-check, not independent origin validation'))
    pd.DataFrame(rows).to_csv(OUT/'national_facility_output_reconciliation.csv',index=False)
    raw=pd.read_csv(BACI)
    subset=raw[raw.year.eq(2018)&raw.hs07_code.eq(750210)&raw.trade_value_usd.gt(0)&
        (raw.exporter_iso.eq('FIN')|raw.importer_iso.eq('FIN'))].copy()
    assert subset.quantity_tonnes.notna().all() and subset.quantity_tonnes.gt(0).all()
    subset['direction']=np.where(subset.exporter_iso.eq('FIN'),'export','import')
    subset['retained_in_previous_model']=subset.exporter_iso.str.fullmatch('[A-Z]{3}')&subset.importer_iso.str.fullmatch('[A-Z]{3}')&subset.exporter_iso.ne(subset.importer_iso)
    subset.to_csv(OUT/'finland_hs750210_2018_record_audit.csv',index=False)
    countries=pd.read_csv(ROOT/'external_sources/baci_hs07_202601/country_codes_V202601.csv')
    special=countries[countries.country_code.eq(490)]
    assert len(special)==1 and special.country_iso3.iloc[0]=='S19'
    assert special.country_name.iloc[0]=='Other Asia, nes'
    special.to_csv(OUT/'unassigned_partner_original_metadata.csv',index=False)
    incoming=subset[subset.direction.eq('import')]
    imported_upper=float(incoming.quantity_tonnes.sum()+.0005*len(incoming))
    own_lower=float(bench.loc[bench.year.eq(2018),'own_russian_feed_output'].iloc[0]-.5)
    national_upper=50435+10330+1.
    unresolved=national_upper-own_lower+imported_upper
    inputs=dict(year=2018,hs07=750210,national_refined_output_upper_t_Ni=national_upper,
        known_own_Russian_feed_output_lower_t_Ni=own_lower,import_product_mass_upper_t=imported_upper,
        unassigned_supply_upper_before_additional_allowance_t_Ni=unresolved,
        purity_floor_from_2017_note=.99-.015,purity_is_assay=False,
        rounding_assumptions=dict(refinery_each_integer_t_half_width=.5,facility_integer_t_half_width=.5,trade_product_t_half_width=.0005),
        bridge_is_conditional=True,additional_allowance_observed=False,
        provenance='operator_own_Russian_feed_not_independently_audited_mine_origin')
    (OUT/'national_bridge_inputs.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    checks,compatibility=[],[]
    fingerprints={}
    for variant in ['original3','expanded6']:
        fp=pd.read_csv(ROOT/f'research_process/nickel_scope_correction_20260908/results/{variant}/exporter_origin_profiles.csv.gz')
        s=fp[fp.year.eq(2018)&fp.stage.eq('metal')&fp.exporter_iso.eq('FIN')&fp.inferred_mine_origin_iso.eq('RUS')]
        assert len(s)==1
        fingerprints[variant]=float(s.share.iloc[0])
    for name,selection in [('all_reported_partners',subset.direction.eq('export')),
        ('previous_model_partner_subset',subset.direction.eq('export')&subset.retained_in_previous_model)]:
        outgoing=subset[selection]
        mass_low=float(outgoing.quantity_tonnes.sum()-.0005*len(outgoing))
        for purity in [.975,1.]:
            nickel_low=mass_low*purity
            for extra in [0.,5000.,10000.,20000.,float('inf')]:
                checks.append(dict(target=name,purity_floor=purity,export_records=len(outgoing),
                    export_product_mass_low_t=mass_low,export_ni_lower_t=nickel_low,
                    unassigned_supply_upper_t=unresolved,additional_allowance_t=extra,
                    russian_feed_export_mass_share_lower=export_lower(nickel_low,unresolved,extra),
                    russian_feed_export_mass_share_upper=1.,
                    inference='conditional_mass_bound_not_export_value_share'))
            for variant,share in fingerprints.items():
                compatibility.append(dict(target=name,purity_floor=purity,model_variant=variant,
                    model_export_value_share=share,
                    minimum_extra_t_if_value_share_equals_nickel_mass_share=max(0.,nickel_low*(1-share)-unresolved),
                    share_equivalence_observed=False,model_not_overwritten=True))
    pd.DataFrame(checks).to_csv(OUT/'national_export_mass_bounds_2018.csv',index=False)
    pd.DataFrame(compatibility).to_csv(OUT/'conditional_model_compatibility_allowance.csv',index=False)
    return inputs

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=source_checks()
    bench=facility()
    inputs=national(bench)
    sherritt=' '.join(html.fromstring((EXT/'sherritt_2024_results.html').read_bytes()).text_content().split())
    assert 'lower third-party feed purchases in 2024' in sherritt and '30,331' in sherritt
    admissions=[dict(evidence='Harjavalta 2012–2021 own-feed output',target='same facility, year and saleable-Ni output',status='accepted_lower_bound',reason='matched operator provenance and contained-metal denominator'),
        dict(evidence='Harjavalta 2018 product split plus own-feed output',target='same facility/year/product Ni output',status='accepted_conditional_product_bound',reason='nonnegative material allocation and stated rounding envelope'),
        dict(evidence='Finnish refinery output plus trade quantities',target='2018 national HS750210 contained-Ni export',status='conditional_only',reason='additional supply allowance, classification, reported-quantity and production-coverage conditions'),
        dict(evidence='Harjavalta facility mass constraints',target='national export-value mine-origin fingerprint',status='not_admitted',reason='entity, product, unit and provenance do not match; no observed bridge'),
        dict(evidence='Sherritt 2024 finished nickel and named Moa feed route',target='100% Cuban origin of refinery output or Canadian exports',status='not_admitted',reason='third-party feed and inventories; ownership share is not origin share'),
        dict(evidence='Harjavalta 2012–2021 series',target='2022–2024 facility origin shares',status='not_admitted',reason='missing years; no extrapolated observations')]
    pd.DataFrame(admissions).to_csv(OUT/'constraint_admission_register.csv',index=False)
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [BACI,FAC/'harjavalta_origin_linked_output_2012_2021.csv',FAC/'nornickel_ar2021_operations.html',ROOT/'external_sources/baci_hs07_202601/country_codes_V202601.csv']}
    (OUT/'run_manifest.json').write_text(json.dumps(dict(source_hashes=source_hashes,archived_new_sources=sum(r['status']=='archived' for r in manifest),
        facility_years=10,independent_facilities=1,product_groups_2018=5,
        national_year_calculated=2018,national_origin_value_inputs_changed=False,
        fitted_weights=False,conditional_mass_bridge=inputs,code_sha256=sha(__file__)),indent=2),encoding='utf-8')
    print(pd.read_csv(OUT/'facility_product_origin_bounds_2018.csv')[['product','own_feed_share_lower']].to_string(index=False))
    print(pd.read_csv(OUT/'national_export_mass_bounds_2018.csv').query('purity_floor == .975').to_string(index=False))
    print(pd.read_csv(OUT/'conditional_model_compatibility_allowance.csv').query('purity_floor == .975').to_string(index=False))

if __name__=='__main__':
    main()
