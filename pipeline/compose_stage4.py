"""Stage 4 composed config: AdaLN sidecar + Kitchen INT8 (+ optional Sage) on the LightX2V4 baseline.

Composes the retained runner's activation (build_local_config) with the follow-up
feature flags. Driver-level explicit composition; the source-level defaults patch
(stage4-defaults.patch from the composition lane) makes the same settings the
stock-launch default. Control arm uses the current v2 baseline (no AdaLN/Kitchen/Sage).
"""
from __future__ import annotations
import json
from pathlib import Path

def _stage4_kitchen_source_evidence(native,site):
    """Stage 4 rebind of the retained Kitchen gate to the current composition-v2 runtime."""
    import json
    from pathlib import Path
    ROOT=Path('/mnt/Projects/MiniAcc')
    path=ROOT/'stage4/kitchen-current-sources/current-sources.json'
    entries=json.loads(path.read_text())
    dependencies=json.loads((path.with_name('dependencies.json')).read_text())['files']
    package=json.loads((ROOT/'stage3/redo-20260915/kitchen-package-reuse.json').read_text())['files']
    bound=entries+dependencies+package
    for entry in bound:
        rel=Path(entry['path'])
        assert not rel.is_absolute() and '..' not in rel.parts,'unsafe path'
        source=site/rel
        assert source.is_file() and not source.is_symlink() and native._sha256(source)==entry['sha256'],'Kitchen stage4 source mismatch: '+entry['path']
    return bound

def _stage4_adaln_sidecar_evidence(native,site,adapter):
    """Stage 4 rebind of the AdaLN gate: same sidecar/receipt checks, source hashes from the stage4 receipt."""
    import json
    from pathlib import Path
    ROOT=Path('/mnt/Projects/MiniAcc')
    root=ROOT/'stage3/adaln-sidecar-light4-20260915'
    ready_path=root/'ready.json';verification_path=ROOT/'stage3/adaln-sidecar-light4-20260915-native-check/verification.json';sidecar=root/'cache.safetensors'
    for path in (ready_path,verification_path,sidecar):
        assert path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'missing/unsafe AdaLN evidence'
    assert native._sha256(ready_path)=='3012525eae02585662545c6091999cb88d43c1b18ac3bb684f29f95d297d02c6'
    assert native._sha256(verification_path)=='f0b1d12eb2cc02644e471bb224fc7452b854bc41b90344d915beb063ea334c93'
    ready=json.loads(ready_path.read_text());verification=json.loads(verification_path.read_text())
    expected='8f794c6049b8fdfbb85793be29fe79e15ce02a3d00bbd729c2137bdadcdc34d0'
    provenance=ready['provenance']
    checks=[ready.get('status')=='prepared_not_benchmarked',ready.get('native_storage_roundtrip') is True,ready.get('native_lookup_checks')==4,ready.get('sidecar')==str(sidecar),ready.get('sha256')==verification.get('sidecar_sha256')==native._sha256(sidecar)==expected,verification.get('status')=='native_projection_equivalence_passed',verification.get('exact_equal') is True,verification.get('projection_plan_comparisons')==204,verification.get('normalized_adapter_keys')==624,verification.get('cache_dependency_targets')==[],provenance['adapter']['sha256']==adapter['actual_sha256'],provenance['model_revision']==native.SNAPSHOT.name,provenance['plan_gemm_rows']==[1,2,2,2],provenance['tp_size']==1,provenance['matmul_allow_tf32'] is False,verification.get('matmul_allow_tf32') is False]
    stage4=json.loads((ROOT/'stage4/adaln-current-source.json').read_text())['native_source_sha256']
    runtime=site/'sglang/multimodal_gen/runtime'
    checks.extend(native._sha256(runtime/name)==sha for name,sha in stage4.items())
    plans=ROOT/'artifacts/review/stage3-adaln-plan-parent-20260915/native-plans.json'
    checks.append(native._sha256(plans)==provenance['plan_record_sha256'])
    assert all(checks) and not (root/'failure.json').exists(),'AdaLN stage4 gate failed'
    return {'path':str(sidecar),'sha256':expected,'ready':str(ready_path),'verification':str(verification_path),'projection_plan_comparisons':204,'scope':'local RTX4090 native projections; stage4 rebind to composition-v2 source; not full-model or cross-hardware equivalence'}

def compose_stage4_config(rf,native,arm,purpose):
    native._kitchen_source_evidence=lambda site:_stage4_kitchen_source_evidence(native,site)
    native._adaln_sidecar_evidence=lambda site,adapter:_stage4_adaln_sidecar_evidence(native,site,adapter)
    manifest=native._load_manifest(native.MANIFEST)
    if arm=='control':
        config=rf.build_config('baseline',purpose)
        config['candidate_id']='control'
        return config
    config=native.build_local_config(manifest,'kitchen_int8',purpose)
    site=native._runtime_path()/f'lib/python3.12/site-packages'
    sidecar=native._adaln_sidecar_evidence(site,config['adapter'])
    config['argv']+=['--minimax-h3-adaln-cache-path',sidecar['path']]
    config['adaln_sidecar']=sidecar
    config['env'].update(rf.FLAGS)
    config['env']['CUDA_VISIBLE_DEVICES']='0'
    config['env']['MINIACC_STAGE3_SAGE']='1' if arm=='adaln_kitchen_sage' else '0'
    config['env']['MINIACC_STAGE3_SOL']='1' if arm=='adaln_kitchen_sol' else '0'
    config['env']['MINIACC_STAGE4_ADALN_KITCHEN']='1'
    config['stage4_arm']=arm
    config['candidate_id']={'adaln_kitchen_sage':'sage_qk8_pv16'}.get(arm,arm)
    config['sampler']='euler'
    config['feature_delta']='stage4 hard-coded pipeline: AdaLN sidecar (post-adapter, 204-key bitwise-verified) + Kitchen INT8 weight-only (explicit CUDA, ConvRot256, no fallback)' + (' + SageAttention2.2.0 QK INT8/PV FP16' if arm=='adaln_kitchen_sage' else '')
    manifest4=json.loads((Path('/mnt/Projects/MiniAcc/stage4/deployment-stage4.json')).read_text())
    site4=native._runtime_path()/'lib/python3.12/site-packages'
    rows4=[]
    for row in manifest4['installed']:
        path=site4/row['relative_path']
        if path.is_symlink() or not path.is_file() or native._sha256(path)!=row['sha256']:
            raise RuntimeError('stage4 source admission failed:'+str(path))
        rows4.append({'path':str(path),'sha256':row['sha256']})
    config['feature_sources']=rows4
    config['followup_authority']=str(rf.AUTH/'execution-authority.json')
    config['shared_decoder_residency']={'receipt':'stage3/redo-20260915/residency-parent-gate/deployment-receipt.json','precision':'unchanged native selective decoder placement; FP32 islands preserved'}
    config['host_copy_reserve_accommodation']={'bytes':17179869184,'receipt':'artifacts/review/stage3-host-copy-reserve-20260915/installation.json'}
    rf.CURRENT=config  # follow-up effective-environment validator binds the built config
    return config
