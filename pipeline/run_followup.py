#!/usr/bin/env python3
"""Follow-up adapters around the retained guarded Stage3 caller/lifecycle runner.

The historical runner file is not changed. Only explicit request fields,
source admission, candidate environment, and correlated feature proof extend it.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('retained_stage3_runner',ROOT/'scripts/run_stage3_local.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
ORIGINAL_BODY=native._video_body
ORIGINAL_WARMUP=native._execute_warmup
ORIGINAL_CPU=native._cpu_native_request_validation
ORIGINAL_ENV=native._effective_environment
ORIGINAL_VALIDATE_ENV=native._validate_effective_environment
AUTH=ROOT/'stage3/followup-20260916'
FLAGS={
    'MINIACC_STAGE3_SAGE':'0','MINIACC_STAGE3_VIDEO_QUERY_MERGE_RATIO':'0',
    'MINIACC_STAGE3_SOL':'0','MINIACC_STAGE3_VIDEO_FFN_MERGE_RATIO':'0',
    'MINIACC_STAGE3_VIDIT_Q':'0','SGLANG_CACHE_DIT_ENABLED':'false',
    'SGLANG_ENABLE_TORCH_COMPILE':'false','MINIACC_STAGE3_FOLLOWUP_EVIDENCE':'1',
    'MINIACC_STAGE3_SAGE_TRACE':'0','MINIACC_STAGE3_CAPTURE_ROOT':'',
    'MINIACC_STAGE3_CAPTURE_PROMPT_SHA256':'',
}
CURRENT=None
METHODS={
    'baseline':'current-composition native BF16 Euler/FA control',
    'sage_qk8_pv16':'SageAttention2.2.0 QK INT8 per-thread; FP16 PV/FP32 accumulation; NHD',
    'query_merge_125':'ToMeSD-inspired2x2x2 query-only target-video12.5%; dense KV; native refiners',
    'heun_nfe4':'Heun coupled native indices[0,2,4],two predictor/corrector intervals/four actual calls',
}

def source_evidence():
    manifest=json.loads((AUTH/'deployment-v2.json').read_text())
    site=native._runtime_path()/'lib/python3.12/site-packages'
    rows=[]
    for row in manifest['installed']:
        path=site/row['relative_path']
        if path.is_symlink() or not path.is_file() or native._sha256(path)!=row['sha256']:
            raise RuntimeError('follow-up source admission failed:'+str(path))
        rows.append({'path':str(path),'sha256':row['sha256']})
    return rows

def build_config(method,purpose='speed'):
    global CURRENT
    if method not in METHODS:raise ValueError('unknown follow-up arm:'+method)
    config=native.build_local_config(native._load_manifest(native.MANIFEST),'baseline',purpose)
    config['candidate_id']=method
    config['feature_delta']=METHODS[method]
    config['feature_sources']=source_evidence()
    config['followup_authority']=str(AUTH/'execution-authority.json')
    config['env'].update(FLAGS)
    config['env']['CUDA_VISIBLE_DEVICES']='0'
    config['sampler']='heun' if method=='heun_nfe4' else 'euler'
    if method=='sage_qk8_pv16':config['env']['MINIACC_STAGE3_SAGE']='1'
    if method=='query_merge_125':config['env']['MINIACC_STAGE3_VIDEO_QUERY_MERGE_RATIO']='0.125'
    config['shared_decoder_residency']={'receipt':'stage3/redo-20260915/residency-parent-gate/deployment-receipt.json','precision':'unchanged native selective decoder placement; FP32 islands preserved'}
    config['host_copy_reserve_accommodation']={'bytes':17179869184,'receipt':'artifacts/review/stage3-host-copy-reserve-20260915/installation.json'}
    CURRENT=config
    return config

def video_body(prompt,config):
    body=ORIGINAL_BODY(prompt,config)
    body['minimax_h3_sampler']=config['sampler']
    body['minimax_h3_probe_finite']=bool(config.get('warmup_probe',False))
    return body

def events(output,event_name,request_id):
    found=[]
    for line in (Path(output)/'server.log').read_text(errors='replace').splitlines():
        if event_name+' ' not in line:continue
        try:value=json.loads(line.split(event_name+' ',1)[1])
        except json.JSONDecodeError:continue
        if value.get('request_id')==request_id:found.append(value)
    return found

def feature_proof(config,output,request_id,*,finite=False):
    records=events(output,'stage3_followup_forward',request_id)
    heun=config['sampler']=='heun'
    indices=[0,2,2,4] if heun else [0,1,2,3]
    roles=['predictor','corrector']*2 if heun else ['euler']*4
    video=[1.,.9729729890823364,.9230769276618958,.800000011920929,0.]
    audio=[1.,.8999999761581421,.75,.5,0.]
    checks={
        'actual_four_calls':len(records)==4 and [r.get('forward_index') for r in records]==[0,1,2,3],
        'solver':all(r.get('solver')==config['sampler'] for r in records),
        'roles':[r.get('role') for r in records]==roles,
        'sigma_indices':[r.get('sigma_index') for r in records]==indices,
        'sigmas':len(records)==4 and all(isinstance(r.get('sigma_video'),(int,float)) and isinstance(r.get('sigma_audio'),(int,float)) and math.isclose(r['sigma_video'],video[i],abs_tol=1e-7,rel_tol=0) and math.isclose(r['sigma_audio'],audio[i],abs_tol=1e-7,rel_tol=0) for r,i in zip(records,indices)),
        'excluded_probe_finite':not finite or all(r.get('finite_output') is True for r in records),
        'validation_not_timed':finite or all(r.get('finite_output') is None for r in records),
    }
    sage=config['candidate_id']=='sage_qk8_pv16'
    checks['actual_sage_calls']=all(r.get('sage_main_calls')==(50 if sage else 0) and type(r.get('sage_other_calls')) is int and (r['sage_other_calls']>=0 if sage else r['sage_other_calls']==0) for r in records)
    token=events(output,'stage3_query_merge_request',request_id)
    if config['candidate_id']=='query_merge_125':
        expected={'attention_calls':200,'query_rows_before':7552000,'query_rows_after':6619600,'query_rows_removed':932400,'video_rows_before':7459200,'video_rows_after':6526800,'key_rows':7552000,'value_rows':7552000,'canonical_output_rows':7552000}
        checks['actual_query_rows']=len(token)==1 and all(type(token[0].get(k)) is int and token[0][k]==v for k,v in expected.items())
    else:checks['no_hidden_query_merge']=not token
    return {'kind':config['candidate_id'],'events':records,'query_events':token,'checks':checks,'passed':all(checks.values())}

def dispatch(config,output,request_id,perf_path=None):
    perf_path=perf_path or Path(output)/'perf'/f'{request_id}.json'
    report=json.loads(Path(perf_path).read_text())
    steps=report.get('denoise_steps_ms',[])
    valid=(report.get('request_id')==request_id and len(steps)==4 and all(isinstance(s,dict) and type(s.get('step')) is int and s['step']==i and type(s.get('duration_ms')) in (int,float) and math.isfinite(s['duration_ms']) and s['duration_ms']>0 for i,s in enumerate(steps)))
    admission_path=Path(output)/'warmup-native-admission.json'
    admission=json.loads(admission_path.read_text())
    quiet=(not events(output,'stage3_followup_forward',request_id) and not events(output,'stage3_query_merge_request',request_id))
    linked=(admission.get('binding')==admission_binding(config) and admission.get('proof',{}).get('passed') is True and admission.get('media',{}).get('valid') is True)
    proof={'kind':config['candidate_id'],'passed':quiet and linked,'scope':'timed policy/source/environment linked to excluded native admission; no timed per-kernel trace','optional_evidence_absent':quiet,'admission_path':str(admission_path),'admission_link_valid':linked}
    observed=re.findall(r'Using ([a-z0-9_]+) attention backend',(Path(output)/'server.log').read_text(errors='replace'),re.I)
    # Sage is an explicit H3-only override. FA in Qwen/other components is not a H3 fallback.
    if config['candidate_id']=='sage_qk8_pv16':
        backend_ok='stage3_sage_backend_explicit qk8_pv16_fp32' in (Path(output)/'server.log').read_text(errors='replace')
    else:backend_ok=bool(observed) and all(x.lower()=='fa' for x in observed)
    return {'request_id':request_id,'feature_proof':proof,'selected_backend':'sage_explicit_qk8_pv16' if config['candidate_id']=='sage_qk8_pv16' else 'fa','observed_backends':observed,'fallback_detected':not backend_ok,'perf_report_path':str(perf_path),'perf_report':report,'perf_report_error':None,'request_identity_match':report.get('request_id')==request_id,'forward_count':len(steps) if valid else 0,'forward_records':steps,'status':'passed' if valid and proof['passed'] and backend_ok else 'blocked_missing_native_evidence'}

def admission_binding(config):
    data={key:config[key] for key in ('candidate_id','sampler','workload','feature_sources')}
    data['flags']={name:config['env'][name] for name in FLAGS}
    data['deployment_sha256']=native._sha256(AUTH/'deployment-v2.json')
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def warmup(config,prompt,output,base,**kwargs):
    probe=copy.deepcopy(config);probe['warmup_probe']=True
    record=ORIGINAL_WARMUP(probe,prompt,output,base,**kwargs)
    proof=feature_proof(config,output,record['request_id'],finite=True)
    media=native._media_validation(Path(record['output']))
    native.atomic_write_json(Path(output)/'warmup-native-admission.json',{'proof':proof,'media':media,'excluded':True,'binding':admission_binding(config)})
    if not proof['passed'] or not media.get('valid'):raise RuntimeError('excluded warmup failed native feature/finite/AV admission')
    return record

def cpu_validate(payload):
    result=ORIGINAL_CPU(payload)
    from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import VideoGenerationsRequest
    from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import _video_request_model_kwargs
    request=VideoGenerationsRequest(**payload['body'])
    kwargs={**result['lowered'],**_video_request_model_kwargs(request,MiniMaxH3SamplingParams)}
    lowered=MiniMaxH3SamplingParams.lower_video_request_kwargs(request,kwargs)
    actual=MiniMaxH3SamplingParams(**lowered)
    expected=payload['body'].get('minimax_h3_sampler','euler')
    probe=payload['body'].get('minimax_h3_probe_finite',False)
    if actual.minimax_h3_sampler!=expected or actual.minimax_h3_probe_finite is not probe:raise RuntimeError('native sampler/probe lowering mismatch')
    result['lowered']=lowered
    result['checks']['followup_native_lowering']=True
    result['followup']={'sampler':expected,'probe_finite':probe,'actual_nfe':4,'sigma_indices':[0,2,2,4] if expected=='heun' else [0,1,2,3]}
    return result

def effective_env(pid):
    values=ORIGINAL_ENV(pid)
    for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
        if b'=' not in item:continue
        key,value=item.split(b'=',1)
        if key.decode() in FLAGS:values[key.decode()]=value.decode()
    return values

def validate_env(values,runtime):
    ORIGINAL_VALIDATE_ENV(values,runtime)
    if CURRENT is None:raise RuntimeError('follow-up environment lacks built config')
    for name in FLAGS:
        if values.get(name)!=CURRENT['env'][name]:raise RuntimeError('effective follow-up flag mismatch:'+name)

# Deliberate, bounded adapter points; lifecycle/timing/guards/cleanup stay retained.
native._video_body=video_body
native._execute_warmup=warmup
native._dispatch_evidence=dispatch
native._effective_environment=effective_env
native._validate_effective_environment=validate_env
native.__file__=__file__ # Isolated request validation re-enters this explicit adapter.

def main():
    if sys.argv[1:]==['--cpu-validate-request']:
        print(json.dumps(cpu_validate(json.loads(sys.stdin.read())),allow_nan=False));return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=float,default=7200)
    args=parser.parse_args()
    if not __debug__:raise RuntimeError('optimized Python is forbidden')
    config=build_config(args.method)
    if not args.execute:print(json.dumps(config,indent=2));return 0
    from datetime import datetime,timezone
    if datetime.now(timezone.utc)>=datetime(2026,9,16,14,tzinfo=timezone.utc):raise RuntimeError('follow-up generation launch cutoff passed')
    return native.execute_request_campaign(config,args.output,timeout_seconds=args.timeout)

if __name__=='__main__':raise SystemExit(main())
