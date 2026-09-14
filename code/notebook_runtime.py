"""Execute notebook code cells and resolve local notebook imports."""
from pathlib import Path
import importlib.abc, importlib.util, json, sys

def code_cells(path):
    doc=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return [''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code']

class NotebookLoader(importlib.abc.Loader):
    def __init__(self,path):self.path=Path(path)
    def create_module(self,spec):return None
    def exec_module(self,module):
        module.__file__=str(self.path)
        cells=code_cells(self.path)
        for text in cells[1:]:exec(compile(text,str(self.path),'exec'),module.__dict__)

class NotebookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        name=fullname.rsplit('.',1)[-1]
        for folder in path if path is not None else sys.path:
            candidate=Path(folder)/f'{name}.ipynb'
            if candidate.is_file():return importlib.util.spec_from_loader(fullname,NotebookLoader(candidate),origin=str(candidate))

def configure(namespace,relative):
    root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'tools/notebook_runtime.py').is_file() or (p/'code/notebook_runtime.py').is_file())
    path=root/relative
    if not path.is_file() and (root/'environment/code_files.json').exists():
        records=json.loads((root/'environment/code_files.json').read_text(encoding='utf-8'))
        match=next((r for r in records if r.get('runtime_path')==relative),None)
        if match:path=root/match['path']
    namespace['__file__']=str(path)
    sys.argv=[str(path)]
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    if not any(isinstance(x,NotebookFinder) for x in sys.meta_path):sys.meta_path.insert(0,NotebookFinder())

def execute(path):
    path=Path(path).resolve()
    namespace={'__name__':'__main__','__file__':str(path)}
    sys.path.insert(0,str(path.parent))
    for text in code_cells(path):exec(compile(text,str(path),'exec'),namespace)
    return namespace

if __name__=='__main__':execute(sys.argv[1])
