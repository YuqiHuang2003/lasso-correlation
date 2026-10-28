from pathlib import Path
from fractions import Fraction
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_column_permutation_v1 import score_range,event_state,block_state,qc_limited_state,continued_methods


def test_exact_f1_boundary_is_not_a_strict_event():
    scores=[dict(tp=0,fp=0,fn=1)]*5+[dict(tp=1,fp=98,fn=0)]
    result=score_range(scores,[.02,.01,0.])
    assert Fraction(result['range_fraction_numerator'],result['range_fraction_denominator'])==Fraction(1,50)
    assert result['f1_event_states']['0.02']=='no_event'
    assert result['f1_event_states']['0.01']=='event'


def test_partial_agreement_is_unresolved_but_observed_event_survives():
    assert event_state(4,6,False)=='undecidable'
    assert event_state(2,6,True)=='event'
    assert block_state(['no_event']*3,4)=='undecidable'
    assert block_state(['event','undecidable'],4)=='event'


def test_repeat_qc_does_not_erase_raw_event_or_certify_it():
    raw='event'
    assert qc_limited_state(raw,False,True)=='undecidable'
    assert raw=='event'
    assert qc_limited_state(raw,True,True)=='event'


def test_different_methods_do_not_supply_same_method_repetition():
    rows=[dict(source='uji',seed=300,method='sklearn',tau=1e-6,metric='support',layer='qc',state='event'),
          dict(source='uji',seed=301,method='skglm',tau=1e-6,metric='support',layer='qc',state='event')]
    assert continued_methods(rows)==[]
    rows.append(dict(rows[0],seed=302))
    assert continued_methods(rows)==['sklearn']
