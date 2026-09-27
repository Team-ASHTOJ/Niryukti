"""Measured RHS response, not a differentiable layer or basis validity range."""
import copy
import math
from .planning import SolverSession

def rhs_sensitivity(model, row, delta, *, session_type=SolverSession, **solve_options):
    """Shift both finite bounds of one row by +/-delta and re-solve.

    Reports two-sided secants only when all three solves report optimality.
    Integer problems are deliberately excluded: no smooth derivative is implied.
    """
    data=copy.deepcopy(model.data if hasattr(model,'data') else model)
    if not math.isfinite(delta) or delta<=0: raise ValueError('delta must be finite and positive')
    if any(v.get('type','continuous')!='continuous' for v in data['variables']):
        raise ValueError('RHS sensitivity supports continuous models only')
    rows=[r for r in data['constraints'] if r['name']==row]
    if len(rows)!=1:raise ValueError('Expected one uniquely named row')
    bounds={side:rows[0][side] for side in ('lb','ub') if rows[0].get(side) is not None}
    if not bounds:raise ValueError('Row has no finite bound')
    results={}
    with session_type(data) as session:
        results['baseline']=session.solve(**solve_options)
        for name,shift in [('minus',-delta),('plus',delta)]:
            session.update_row_bounds({row:{side:value+shift for side,value in bounds.items()}})
            results[name]=session.solve(**solve_options)
    valid=all(r.get('status') in ('OPTIMAL','VERIFIED_OPTIMAL') and r.get('objective') is not None for r in results.values())
    slopes={}
    if valid:
        base=results['baseline']['objective']
        slopes=dict(left=(base-results['minus']['objective'])/delta,
                    right=(results['plus']['objective']-base)/delta)
    return dict(row=row,delta=delta,status='MEASURED' if valid else 'UNRESOLVED',slopes=slopes,results=results,
                scope='Finite perturbation response in original objective units; not an exact derivative or allowable RHS range.')
