"""Resume unchanged reference inputs with verified equivalent conic solves."""
import argparse
import json
import run_unified as u
import solver_verified as v

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    evidence=dict(solver_code_sha256=u.s.sha(v.__file__),clarabel_version=v.clarabel.__version__,
                  clarabel_binary_sha256={str(x.name):u.s.sha(x) for x in u.Path(v.clarabel.__file__).parent.glob('*.pyd')},
                  reason='OSQP did not converge for Copper 2023 ore; identical QP and acceptance thresholds retained',
                  documentation='https://clarabel.org/stable/python/getting_started_py/')
    (u.OUT/f"solver_amendment_{evidence['solver_code_sha256'][:12]}.json").write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    u.s.solve=v.solve
    if args.pilot:
        import run_allocation_sensitivity_pilot as pilot
        # Keep the interrupted OSQP pilot and its byte-level input contract intact.
        pilot.PILOT=u.OUT/'nickel_joint_allocation_pilot_conic'
        pilot.main()
    else:
        # Reuse frozen derived inputs; do not rewrite gzip headers during resume.
        contract=json.loads((u.OUT/'input_contract.json').read_text())
        for path,digest in contract['inputs'].items():
            assert u.s.sha(u.ROOT/path)==digest
        assert u.s.sha(u.__file__)==contract['driver_sha256']
        trade=u.pd.read_csv(u.OUT/'trade_corridors.csv.gz')
        infer,lm,core=u.modules()
        annual=u.pd.read_csv(u.OUT/'annual_indicators.csv.gz')
        original_select=core.local.rolling_universe
        # The legacy pandas agg fails on an empty single-mineral frame. Select
        # the full independent-unit universe first, then retain this mineral.
        cohorts={y:original_select(annual,y) for y in range(2012,2025)}
        def selected(frame,year):
            return cohorts[year][cohorts[year].metal.isin(frame.metal.unique())].copy()
        core.local.rolling_universe=selected
        u.allocate(trade,annual,core,[2024]+list(range(2012,2024)))
        print('All requested reference allocations completed.',flush=True)
