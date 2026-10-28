from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_uji_preprocessing_control_v1 import (exact_groups,describe_groups,aggregate,mapping_diagnostics,
                                             target_parameters,truth_identity_diagnostics,qualification)


def test_raw_exact_groups_keep_earliest_but_final_float_bytes_need_not_match():
    raw=np.array([[1.,1.,2.],[3.,3.,4.],[5.,5.,6.]])
    assert exact_groups(raw)==[[0,1],[2]]
    final=raw.copy();final[:,1]+=1e-14
    assert exact_groups(final)==[[0],[1],[2]]
    groups=describe_groups(final,exact_groups(raw))
    assert groups[0]['anchor_column_0based']==0
    assert len(groups[0]['member_scales_from_anchor'])==2
    assert groups[0]['residual_l2_norms'][1]>0


def test_mapping_signal_and_l1_cancellation_are_accounted_separately():
    first=np.array([-1.,0.,1.,2.]);X=np.column_stack([first,-first,np.array([2.,1.,0.,1.])])
    groups=describe_groups(X,[[0,1],[2]])
    weights=np.array([1.,1.,.5]);theta=aggregate(groups,weights)
    np.testing.assert_allclose(theta,[0.,.5],atol=1e-14)
    result=mapping_diagnostics(X,X@weights,groups,weights,.1)
    assert result['signal_or_fit_mapping_rmse']<1e-14
    assert abs(result['l1_cancellation_amount']-2)<1e-14
    assert result['l1_change_lower_bound']-1e-14<=result['l1_norm_change']<=result['l1_change_upper_bound']+1e-14
    assert abs(result['l1_accounting_residual'])<1e-14


def test_nonunit_scale_mapping_has_explicit_representation_and_penalty_bounds():
    x=np.array([-2.,-1.,1.,2.]);X=np.column_stack([x,1.01*x+np.array([.001,0,0,-.001])])
    groups=describe_groups(X,[[0,1]])
    weights=np.array([1.,-.2]);result=mapping_diagnostics(X,np.zeros(4),groups,weights,.3)
    assert result['signal_or_fit_mapping_rmse']<=result['mapping_rmse_triangle_bound']+1e-14
    assert abs(result['penalty_change'])<=.3*result['absolute_l1_change_upper_bound']+1e-14


def test_posthoc_grid_and_public_call_cap():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/uji_preprocessing_control_v1.json').read_text())
    assert len(target_parameters(config))==12
    assert config['maximum_new_public_reference_calls']==48
    assert set(config['selected_seeds'])=={300,305,308}


def test_identity_and_cancellation_flags_are_json_serializable_python_scalars():
    X=np.column_stack([np.array([-1.,0.,1.]),np.array([1.,0.,-1.]),np.array([0.,1.,0.])])
    groups=describe_groups(X,[[0,1],[2]])
    result=truth_identity_diagnostics(groups,np.array([1.,1.,0.]),[1e-6,1e-4])
    encoded=json.dumps(result,allow_nan=False)
    assert encoded
    for row in result.values():
        assert row['groups_with_any_true_variable']==1
        assert row['groups_containing_multiple_true_variables']==1
        assert row['true_variable_identities_merged_count']==1
        assert row['groups_with_true_variables_but_net_below_tau']==1
        assert row['deleted_columns_with_true_nonzero_coefficient']==1
        assert all(type(value) is int for value in row.values())


def test_reference_accuracy_is_not_filtered_by_coefficient_consensus():
    config={'references':{'normalized_gap_tol':1e-10,'time_limit':20.,'coef_linf_agreement_tol':1e-7,
                          'normalized_objective_agreement_tol':1e-9},'support_thresholds':[1e-6,1e-4]}
    cert={'primal':1.,'dual':1.-1e-12,'gap':1e-12,'normalized_gap':1e-12,'dual_feasibility':1.}
    refs={'sklearn':{'certificate':cert,'elapsed_sec':.1,'status':'converged'},
          'skglm':{'certificate':cert,'elapsed_sec':.2,'status':'converged'},
          'reference_accepted':False,'coefficient_linf_difference':.1,'normalized_primal_difference':0.,
          'by_support_threshold':{str(tau):{'same_support':False,'same_thresholded_signs':False} for tau in config['support_thresholds']}}
    result=qualification(refs,config)
    assert result['accuracy_eligible'] is True
    assert result['reference_consensus'] is False
    assert not result['qualification_failure_reasons']['accuracy']
    assert 'coefficient_linf_disagreement' in result['qualification_failure_reasons']['consensus']
    json.dumps(result,allow_nan=False)
