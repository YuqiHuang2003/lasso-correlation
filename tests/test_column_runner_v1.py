"""Small arithmetic/mocked-call checks; no optimizer or external response fits."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_column_permutation_v1 as runner
from lasso_study.solvers import certificate


def config():
    return json.loads((ROOT/"configs/column_permutation_v1.json").read_text())


def test_schedule_has_distinct_primary_and_repeat_calls_in_registered_interleaving():
    settings=config();calls=runner.build_manifest(settings)
    assert len(calls)==784
    assert sum(row['is_primary_order'] for row in calls)==672
    assert sum(row['is_identity_repeat'] for row in calls)==112
    assert len({row['call_id'] for row in calls})==784
    assert [(row['source'],row['method']) for row in calls[:6]]==[
        ('uji','sklearn'),('uji','skglm'),('news','sklearn'),('news','skglm'),('gene','sklearn'),('gene','skglm')]
    assert calls[112]['source']=='news'
    assert calls[112]['order_id']=='identity_repeat'
    assert calls[112]['method']=='skglm'
    assert all(row['permutation_order_id']=='identity' for row in calls if row['is_identity_repeat'])
    json.dumps(calls,allow_nan=False)


def test_inverse_mapping_and_two_coordinate_certificates_preserve_one_output():
    rng=np.random.default_rng(8)
    X=np.asfortranarray(rng.normal(size=(12,5)));y=rng.normal(size=12)
    permutation=np.array([3,1,4,0,2]);w=np.array([.1,-.3,0.,1.,-.25])
    original=runner.original_coefficients(w,permutation)
    assert np.array_equal(original[permutation],w)
    first=certificate(X[:,permutation],y,w,.2);second=certificate(X,y,original,.2)
    result=runner.compare_invariance(X,X[:,permutation],w,original,permutation,first,second)
    assert result['passed'] is True
    broken=deepcopy(second);broken['primal']+=1e-4
    assert not runner.compare_invariance(X,X[:,permutation],w,original,permutation,first,broken)['passed']


def test_qualification_separates_precision_budget_and_status_without_gap_slack():
    settings=config()['references'];cert=dict(primal=1.,dual=1.,gap=0.,raw_gap=0.,normalized_gap=1e-10,dual_feasibility=1.,kkt_inf=0.)
    result=runner.independent_qualification(np.zeros(2),cert,'converged',20.,settings)
    assert result['accuracy_eligible'] is True
    late=runner.independent_qualification(np.zeros(2),cert,'budget_exceeded',20.01,settings)
    assert late['precision_eligible'] is True and late['budget_eligible'] is False
    assert late['library_status_eligible'] is False and late['accuracy_eligible'] is False
    above=deepcopy(cert);above['normalized_gap']=np.nextafter(1e-10,np.inf)
    assert not runner.independent_qualification(np.zeros(2),above,'converged',1.,settings)['precision_eligible']


def test_historical_qc_requires_support_even_when_coefficient_delta_is_small():
    settings=config();X=np.eye(2);fresh=np.array([1.01e-6,0.]);old=np.array([.99e-6,0.])
    cert=dict(primal=1.,dual=1.,gap=0.,normalized_gap=0.,dual_feasibility=1.,kkt_inf=0.)
    result=runner.compare_outputs(X,fresh,old,cert,cert,settings)
    assert result['coefficient_linf_difference']<settings['references']['coef_linf_agreement_tol']
    assert result['passed'] is False
    assert not result['by_support_threshold']['1e-06']['same_support']


def test_f1_uses_original_coordinates_and_exact_integer_fraction():
    truth=np.array([1.,0.,1.,0.]);coef=np.array([1.,1.,0.,0.])
    result=runner.score(coef,truth,1e-6)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert result['f1_exact']=={'numerator':1,'denominator':2}
    permutation=np.array([2,0,3,1])
    mapped=runner.original_coefficients(coef[permutation],permutation)
    assert runner.score(mapped,truth,1e-6)==result


@pytest.mark.parametrize('method',['sklearn','skglm'])
def test_public_call_wrapper_with_mocked_library_preserves_certificates(method,monkeypatch):
    settings=config();X=np.asfortranarray(np.sqrt(3)*np.eye(3));truth=np.array([1.,-.5,0.]);y=X@truth
    data={'X':X,'y':y,'w_true':truth,'metadata':{'lambda':.1}}
    order={'permutation':[2,0,1]};expected=np.array([0.,.9,-.4])
    class FakeLasso:
        def __init__(self,**kwargs):
            assert kwargs['warm_start'] is False and kwargs['selection']=='cyclic'
        def fit(self,Xp,yp):
            assert Xp.flags.f_contiguous and np.array_equal(yp,y)
            self.coef_=expected.copy();self.n_iter_=1;return self
    def fake_modern(Xp,yp,lam,**kwargs):
        assert Xp.flags.f_contiguous and lam==.1
        return {'coef':expected.copy(),'status':'converged','attempt_count':1,'attempts':[{'attempt':1}]}
    monkeypatch.setattr(runner,'Lasso',FakeLasso)
    monkeypatch.setattr(runner,'solve_modern',fake_modern)
    result=runner.run_single_method(data,order,method,settings)
    np.testing.assert_allclose(result['coef_original'],[.9,-.4,0.])
    assert result['accuracy_eligible'] is True
    assert result['invariance']['passed'] is True
    assert result['certificate']['normalized_gap']<=1e-10
    assert result['actual_library_fit_attempts']==1


def test_prepare_only_uses_tiny_fake_designs_and_never_warms_or_fits(tmp_path,monkeypatch):
    from lasso_study.column_permutation_v1 import make_permutations,target_id as data_target_id
    settings=config();params=runner.target_parameters(settings)
    (tmp_path/'docs').mkdir()
    (tmp_path/settings['protocol_path']).write_text('fake frozen test protocol')
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(settings))
    indices={runner.target_id(row):i for i,row in enumerate(params)}
    def make_fake_target(config,row,sources):
        design_index=indices[runner.target_id(row)]%13
        X=np.asfortranarray(np.eye(4)*(1.+design_index*.01));y=np.arange(4,dtype=float);truth=np.array([1.,0.,0.,0.])
        hashes={key:runner.array_sha256(value) for key,value in [('X',X),('y',y),('w_true',truth)]}
        return {'X':X,'y':y,'w_true':truth,'metadata':{'hashes':hashes,'lambda':.1},
                'origin':{'existing_grouping_path':None},'historical_reference':{}}
    def forbidden(*args,**kwargs):
        raise AssertionError('Preparation must not warm or fit a solver')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'target_parameters',lambda _:params)
    monkeypatch.setattr(runner,'column_signature',lambda _:('test-signature',{'config_sha256':'config-hash','manifest_sha256':'manifest-hash'}))
    monkeypatch.setattr(runner,'_data_api',lambda:SimpleNamespace(load_sources=lambda _:None,make_target=make_fake_target,
                                                               make_permutations=make_permutations,target_id=data_target_id))
    monkeypatch.setattr(runner,'warmup_modern',forbidden)
    monkeypatch.setattr(runner,'run_single_method',forbidden)
    monkeypatch.setattr(sys,'argv',['run_column_permutation_v1.py','--config',str(config_path),'--prepare-only'])
    runner.main()
    output=tmp_path/settings['output_path'];prepared_path=output/'preparation.json'
    before=runner.sha(prepared_path)
    prepared=json.loads(prepared_path.read_text())
    assert prepared['new_fits']==0 and prepared['warmup_performed'] is False
    assert len(list((output/'datasets').glob('*.json')))==56
    assert len(list((output/'permutations').glob('*.json')))==13
    assert len(list((output/'groups').glob('*.json')))==13
    assert not (output/'runs').exists() and not (output/'reservations').exists()
    runner.main()
    assert runner.sha(prepared_path)==before
    runner.validate_preparation(output,'test-signature','config-hash')
    invocations=[json.loads(path.read_text()) for path in (output/'invocations').glob('*.json')]
    assert len(invocations)==2 and all(row['state']=='finished' for row in invocations)
    assert all(row['new_public_call_slots_reserved']==0 for row in invocations)
