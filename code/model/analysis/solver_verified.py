"""Equivalent conic QP solve, with unscaled primal/dual KKT verification.

Interface basis: https://clarabel.org/stable/python/getting_started_py/
No objectives, feasible sets or acceptance thresholds are changed.
"""
import json
import sys
import run_unified as u
sys.path.insert(0,str(u.ROOT/'research_process/runtime_support/clarabel_scope_vendor'))
import clarabel
import numpy as np
from scipy.sparse import csc_matrix, vstack, eye, triu, diags
ORIGINAL_SOLVE = u.s.solve

def solve(core,model,objective):
    if not model.variable_count: return ORIGINAL_SOLVE(core,model,objective)
    h,c=u.s.matched_quadratic(core,model,objective)
    a=vstack([model.equality.A,model.inequality.A,eye(model.variable_count)],format='csr')
    lb=np.concatenate([model.equality.lb,model.inequality.lb,model.lower])
    ub=np.concatenate([model.equality.ub,model.inequality.ub,model.upper])
    eq=np.isfinite(lb)&np.isfinite(ub)&(lb==ub)
    upper=np.isfinite(ub)&~eq
    lower=np.isfinite(lb)&~eq
    ac=vstack([a[eq],a[upper],-a[lower]],format='csc')
    b=np.concatenate([ub[eq],ub[upper],-lb[lower]])
    ne=int(eq.sum())
    cones=[clarabel.ZeroConeT(ne),clarabel.NonnegativeConeT(len(b)-ne)]
    attempts=[]
    for scaled in [True,False]:
        scale=1/np.maximum(np.asarray(abs(ac).max(axis=1).toarray()).ravel(),1.) if scaled else np.ones(len(b))
        settings=clarabel.DefaultSettings()
        settings.verbose=False
        settings.max_iter=500
        settings.tol_gap_abs=1e-11
        settings.tol_gap_rel=1e-11
        settings.tol_feas=1e-11
        solution=clarabel.DefaultSolver(triu(h).tocsc(),c,(diags(scale)@ac).tocsc(),b*scale,cones,settings).solve()
        x=np.asarray(solution.x)
        z=np.asarray(solution.z)*scale
        vector=np.clip(x,model.lower,model.upper)
        audit=core.constraint_violations(model,vector)
        stationarity=float(abs(h@x+c+ac.T@z).max(initial=0.))
        slack=b-ac@x
        complementarity=float(abs(z[ne:]*slack[ne:]).max(initial=0.))
        sign=float(np.maximum(-z[ne:],0).max(initial=0.))
        good=(str(solution.status)=='Solved' and audit['max_abs_demand_share_error']<=2e-6
              and audit['max_scaled_inequality_violation']<=2e-7 and audit['max_bound_violation']<=2e-8
              and max(stationarity,complementarity,sign)<2e-7)
        record=dict(success=bool(good),message=str(solution.status),iterations=int(solution.iterations),solver='Clarabel '+clarabel.__version__,
                    kkt_stationarity=stationarity,kkt_complementarity=complementarity,dual_sign_error=sign,
                    row_scaling=scaled,objective_value=float(.5*vector@(h@vector)+c@vector),**audit)
        attempts.append(record)
        if good: break
    if not record['success']:
        # Do not accept AlmostSolved by relabelling it. Ask the original QP
        # solver for a separately accepted solution with the same KKT gate.
        vector,fallback=ORIGINAL_SOLVE(core,model,objective)
        fallback['prior_conic_attempts_json']=json.dumps(attempts)
        return vector,fallback
    return vector,dict(record,attempts_json=json.dumps(attempts))

def test():
    _,_,core=u.modules()
    trade=u.pd.read_csv(u.OUT/'trade_corridors.csv.gz').query('metal == "Copper"')
    annual=u.pd.read_csv(u.OUT/'minerals/Copper/annual.csv.gz')
    fps=u.s.fingerprints_dict(u.pd.read_csv(u.OUT/'minerals/Copper/profiles.csv.gz'))
    units=core.local.rolling_universe(annual,2023).query('stage == "ore"')
    baseline,capacity,pools=core.local.year_trade_components(trade,2023)
    model=core.build_group_model(2023,units,baseline,capacity,pools,fps[2023],core.local.load_routes())
    records=[]
    for objective in ['matched_direct','integrated']:
        _,audit=solve(core,model,objective)
        print(objective,json.dumps(audit),flush=True)
        assert audit['success']
        records.append(dict(objective=objective,**audit))
    u.pd.DataFrame(records).to_csv(u.OUT/'copper_2023_conic_solver_test.csv',index=False)
    previous=u.pd.read_csv(u.OUT/'minerals/Copper/allocation/2023_ore_failed.csv')
    old=float(previous.loc[previous.scenario.eq('direct_partner'),'objective_value'].iloc[0])
    assert abs(records[0]['objective_value']-old)<1e-7

if __name__=='__main__': test()
