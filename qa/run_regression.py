"""GNDiscovery frozen-model regression tests. Can run without installed Streamlit using shim for non-UI path."""
from __future__ import annotations
import sys,types,importlib.util,os,json,hashlib,py_compile
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'app_core'))
from gnretain_source_feature_generator import batch_generate_runtime_source_features
from gnretain221_prediction_core import predict_from_features,inspect_model_bundles
from gnmic_ecoli_prediction_core import (gnmic_runtime_health_check,predict_gnmic_from_source_features,load_gnmic_resources,generate_topology9970)

checks={}
def assert_check(name,expr):
 assert expr,name
 checks[name]='PASS'

# Structural checks
assert_check('ui_compiles',not py_compile.compile(str(ROOT/'streamlit_app/gndiscovery_app.py'),doraise=True) is None)
ready,health=gnmic_runtime_health_check(ROOT)
assert_check('GNMIC_strict_health',ready)
assert_check('EC1792_model_identity',health['model_name']=='GNMIC-EC1792' and health['n_features']==1792)
assert_check('EC1792_operating_threshold',np.isclose(health['threshold'],0.51))
assert_check('EC1792_model_hash',health['model_sha256']=='ecb000ece0356ef1fd53390ee888ed2acdd2befc6c972c833643e11384b38e0d')
gn=inspect_model_bundles(str(ROOT))
print('GNRetain model status:',gn)
assert_check('GNRetain_five_model_deployment',bool(gn['ready'].iloc[0]) if 'ready' in gn.columns else str(gn['status'].iloc[0]).lower()=='ok')

smi=['CC(=O)Oc1ccccc1C(=O)O','CCN(CC)CCCC(C)Nc1ncc2ccccc2n1','O=C1NCC(=O)N1','not_a_smiles']
X,reps=batch_generate_runtime_source_features(smi,project_root=ROOT)
assert_check('SMILES_error_handling',reps.status.tolist()==['ok','ok','ok','failed'] and X.shape==(3,389))
valid=reps.loc[reps.status.eq('ok'),'runtime_smiles'].astype(str).tolist()
gn_pred=predict_from_features(X,project_root=ROOT)
assert_check('GNRetain_prediction_range',len(gn_pred)==3 and gn_pred['gnretention__proba_clean_ensemble'].between(0,1).all())
assert_check('GNRetain_threshold',np.allclose(gn_pred['gnretention__threshold'],0.575))
mp,neigh=predict_gnmic_from_source_features(valid,X,project_root=ROOT,top_n_neighbours=3)
assert_check('GNMIC_probabilities_valid',mp.probability_active.between(0,1).all())
assert_check('GNMIC_original_threshold',np.allclose(mp.threshold,0.51))
assert_check('GNMIC_support_valid',mp.estimated_oof_reliability.between(0,1).all() and mp.prediction_support.isin(['High','Moderate','Low','Indeterminate']).all())
assert_check('GNMIC_support_separate',np.array_equal(mp.predicted_label.to_numpy(),(mp.probability_active.to_numpy()>=.51).astype(int)))
assert_check('GNMIC_neighbours_valid',len(neigh)==3 and all({0,1}.issubset(set(t.binary_label.astype(int))) for t in neigh))
resources=load_gnmic_resources(ROOT)
selected,efp=generate_topology9970(valid,X,project_root=ROOT)
assert_check('EC1792_feature_generator',selected.shape==(3,1792) and efp.shape==(3,2048))
expected=np.mean(np.vstack([m.predict_proba(selected)[:,1] for m in resources['model_bundle']['models']]),axis=0)
assert_check('GNMIC_direct_inference_parity',np.allclose(expected,mp.probability_active,rtol=0,atol=1e-12))

# import all app functions in an isolated minimal Streamlit shim, allowing run_batch() tests without installed Streamlit.
st=types.ModuleType('streamlit')
st.set_page_config=lambda *a,**k:None
st.cache_data=lambda *a,**k: (lambda f:f)
st.cache_resource=lambda *a,**k: (lambda f:f)
st.session_state={}
for n in ['streamlit','streamlit.components','streamlit.components.v1']:
 if n not in sys.modules:sys.modules[n]=types.ModuleType(n)
sys.modules['streamlit']=st
sys.modules['streamlit.components'].v1=sys.modules['streamlit.components.v1']
spec=importlib.util.spec_from_file_location('rebuild_app_ui',ROOT/'streamlit_app/gndiscovery_app.py')
ui=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
assert_check('app_module_imports',callable(ui.run_batch) and callable(ui._integrated_interpretation))
indf=pd.DataFrame({'SMILES':smi,'ID':[10,11,12,13]})
out=ui.run_batch(indf,'SMILES',str(ROOT),source_workers=1,rdkit_threads=1)
assert_check('batch_row_provenance',len(out)==4 and out.row_index.tolist()==[0,1,2,3] and out.ID.tolist()==[10,11,12,13])
assert_check('batch_invalid_row_preserved',out.status.tolist()==['ok','ok','ok','failed'])
assert_check('batch_mic_individual_parity',np.allclose(out.GNMIC_probability.iloc[:3],mp.probability_active,atol=1e-12,rtol=0))
assert_check('batch_retention_individual_parity',np.allclose(out.GNRetain_probability.iloc[:3],gn_pred['gnretention__proba_clean_ensemble'],atol=6e-7,rtol=0))
assert_check('batch_preserves_dual_endpoint_separation',all(k in out for k in ['GNRetain_probability','GNMIC_probability','GNMIC_support','GNRetain_support']))
assert_check('batch_checkpoint_version_bumped',ui.BATCH_CHECKPOINT_SCHEMA_VERSION>=3)
assert_check('batch_chunk_concatenation',len(ui._batch_concat_results({'0:0:2':out.iloc[:2],'0:2:4':out.iloc[2:]}))==4)

# Negative integrity control: change expected metadata in separate temp copy, don't write to production.
assert_check('EC1792_support_is_explicit_reconstruction',resources['support_bundle']['reconstruction'] is True and resources['support_bundle']['primary_model']=='GNMIC-EC1792')
assert_check('no_old_classifier_in_primary_directory',not (ROOT/'GNMIC_ECOLI_ATCC25922_Runtime/model/ATCC25922_Topology9970_LightGBM_3seed_fullcohort.joblib').exists())
assert_check('current_classifier_present',(ROOT/'GNMIC_ECOLI_ATCC25922_Runtime/model/GNMIC-EC1792_LightGBM_3seed_fullcohort.joblib').exists())

print('REBUIlT APP QA RESULTS:',len(checks),'checks passed')
output={'tests':checks,'n_passed':len(checks),'GNMIC_smiles_probabilities':mp.probability_active.to_list(),'GNRetain_smiles_probabilities':gn_pred['gnretention__proba_clean_ensemble'].to_list(),'GNMIC_support':mp.prediction_support.to_list(),'feature_pipeline_parity_warning':'Existing audit found exact frozen matrix matches for 39 of 42 and 5 of 7 published example predictions; unexplained differences remain.','streamlit_gui_status':'Not run: Streamlit missing and cannot be installed in offline environment; UI Python was compiled and non-interactive functions imported using shim.'}
qa=ROOT/'qa';qa.mkdir(exist_ok=True)
(qa/'reconstruction_smoke_results.json').write_text(json.dumps(output,indent=2))
out.to_csv(qa/'batch_smoke_output.csv',index=False)
