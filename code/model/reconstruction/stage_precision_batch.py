"""Versioned full paired-input recomputation; no acceptance-tolerance relaxation."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;CAP=HERE/'capsule'
code=(CAP/'code/allocation_verified.py').read_text();assert code.count('=1e-11')==3
(CAP/'code/allocation_precision.py').write_text(code.replace('=1e-11','=1e-13'),encoding='utf-8')
source=(CAP/'run_allocation_conditional.py').read_text().split('    # All expected numerical results are opened after')[0]
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("ARGS=p.parse_args();assert 1<=ARGS.draw<=5","p.add_argument('--branch',choices=['legacy','rebuilt'],required=True);ARGS=p.parse_args();assert 1<=ARGS.draw<=5")
replace("outputs/allocation_conditional/rho_","outputs/allocation_precision/{ARGS.branch}/rho_")
replace("'ALLOCATION_CONDITIONAL_QA.json'","'PRECISION_QA.json'")
replace("HERE/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json'","HERE/'PRECISION_INPUT_MANIFEST.json'")
replace("verified=load('allocation_verified')","verified=load('allocation_precision')")
replace("trade=pd.read_csv(HERE/'outputs/baci/trade_corridors_rebuilt.csv.gz');seed=pd.read_csv(HERE/'outputs/seed/allocation_production_seed_rebuilt.csv.gz')","trade=pd.read_csv(HERE/('expected/SD5_trade_corridors.csv.gz' if ARGS.branch=='legacy' else 'outputs/baci/trade_corridors_rebuilt.csv.gz'));seed=pd.read_csv(HERE/('expected/SD5_production_seeds.csv.gz' if ARGS.branch=='legacy' else 'outputs/seed/allocation_production_seed_rebuilt.csv.gz'))")
replace("print(f'Allocation and ablation complete: {metal}; {count} groups',flush=True)","print(f'Precision {ARGS.branch} {ARGS.draw}/{ARGS.rho}: {metal}; {count} groups',flush=True)")
source=source.split('    # All expected numerical results are opened after')[0]+(HERE/'precision_footer.txt').read_text()
driver=CAP/'run_precision_batch.py';driver.write_text(source,encoding='utf-8');compile(source,str(driver),'exec')
records=[]
for item in json.loads((CAP/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json').read_text()):
    if item['file']=='run_allocation_conditional.py':continue
    records.append(item)
for rel,role in [('code/allocation_precision.py','same solver; stopping tolerance 1e-13'),('run_precision_batch.py','paired branch driver'),('expected/SD5_trade_corridors.csv.gz','legacy input branch only'),('expected/SD5_production_seeds.csv.gz','legacy input branch only')]:
    p=CAP/rel;records.append(dict(file=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role=role))
(CAP/'PRECISION_INPUT_MANIFEST.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print('Full paired precision batch staged')
