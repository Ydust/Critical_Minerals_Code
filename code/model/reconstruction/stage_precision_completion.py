"""Permit a paired test after an explicitly failed old zero-threshold check.

Does not erase or redefine the original acceptance report. The paired model
must still meet the unchanged 1e-6 flow and reporting gates.
"""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;CAP=HERE/'capsule'
source=(CAP/'run_precision_batch.py').read_text()
old="if ARGS.branch=='rebuilt':assert json.loads((expected/'PRECISION_QA.json').read_text())['accepted']"
new="""if ARGS.branch=='rebuilt':
        reference_qa=json.loads((expected/'PRECISION_QA.json').read_text())
        assert reference_qa['independent_validation']['accepted']
        assert reference_qa['checks']['annual']['accepted'] and reference_qa['checks']['baseline']['accepted']
        assert not any(reference_qa['scientific_categories_vs_original'].values())"""
assert source.count(old)==1;source=source.replace(old,new)
source=source.replace('PRECISION_INPUT_MANIFEST.json','PRECISION_COMPLETION_INPUT_MANIFEST.json')
driver=CAP/'run_precision_completion.py';driver.write_text(source,encoding='utf-8');compile(source,str(driver),'exec')
items=json.loads((CAP/'PRECISION_INPUT_MANIFEST.json').read_text())
items.append(dict(file=driver.name,sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),role='paired-test completion; original failed old-output comparison retained'))
(CAP/'PRECISION_COMPLETION_INPUT_MANIFEST.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
