"""Stage 4 campaign drivers: speed (3 arms) and 16-clip quality (2 arms) on the local RTX 4090.

Arms: control (current v2 baseline), adaln_kitchen (hard-coded AdaLN+Kitchen INT8 pipeline),
adaln_kitchen_sage (pipeline + SageAttention). Quality pairing is within-hardware, per-prompt.
Owner deadline 14:00 UTC. Composition binding arrives from the composition lane
(compose_stage4_config); until it is reviewed this driver refuses to execute.
"""
from __future__ import annotations
import importlib.util,json,os,sys,time
from pathlib import Path

ROOT=Path('/mnt/Projects/MiniAcc')
AUTH=ROOT/'stage3/followup-20260916'
STAGE4=ROOT/'stage4'
spec=importlib.util.spec_from_file_location('followup_adapter',AUTH/'run_followup.py')
rf=importlib.util.module_from_spec(spec);sys.modules['followup_adapter']=rf;spec.loader.exec_module(rf)
native=rf.native

COMPOSE=Path('/home/arezy/.cache/miniacc/stage4-20260917/composition/compose_stage4.py')

def load_composer():
    assert COMPOSE.is_file(),'composition lane output not reviewed/deployed yet'
    spec=importlib.util.spec_from_file_location('stage4_compose',COMPOSE)
    m=importlib.util.module_from_spec(spec);sys.modules['stage4_compose']=m;spec.loader.exec_module(m)
    return m

def records16(native_mod,slice_spec):
    original=native_mod._prompt_records
    lo,hi=(int(x) for x in slice_spec.split(':'))
    def _records(purpose):
        if purpose!='quality':return original(purpose)
        data=json.loads((ROOT/'stage1/eval.yaml').read_text())
        assert len(data['prompts'])==16
        return [{'prompt_id':p['id'],'prompt_en':p['prompt_en'],'stratum':p.get('stratum'),'purpose':purpose} for p in data['prompts'][lo:hi]]
    return _records

def quality_dispatch(config,output,request_id,perf_path=None):
    import json,math,re
    perf_path=perf_path or Path(output)/'perf'/f'{request_id}.json'
    report=json.loads(Path(perf_path).read_text())
    steps=report.get('denoise_steps_ms',[])
    valid=(report.get('request_id')==request_id and len(steps)==4 and all(isinstance(x,dict) and type(x.get('step')) is int and x['step']==i and type(x.get('duration_ms')) in (int,float) and math.isfinite(x['duration_ms']) and x['duration_ms']>0 for i,x in enumerate(steps)))
    proof=rf.feature_proof(config,output,request_id,finite=True)
    observed=re.findall(r'Using ([a-z0-9_]+) attention backend',(Path(output)/'server.log').read_text(errors='replace'),re.I)
    arm_name=config.get('stage4_arm',config['candidate_id'])
    if arm_name=='adaln_kitchen_sage':
        backend_ok='stage3_sage_backend_explicit qk8_pv16_fp32' in (Path(output)/'server.log').read_text(errors='replace')
    else:backend_ok=bool(observed) and all(x.lower()=='fa' for x in observed)
    if arm_name=='adaln_kitchen_sol':
        import json as _json
        sol_events=[]
        for line in (Path(output)/'server.log').read_text(errors='replace').splitlines():
            if 'stage3_sol_forward ' in line:
                try:v=_json.loads(line.split('stage3_sol_forward ',1)[1])
                except _json.JSONDecodeError:continue
                if v.get('request_id')==request_id:sol_events.append(v)
        backend_ok=backend_ok and len(sol_events)==4 and [e.get('forward_index') for e in sol_events]==[0,1,2,3] and [e.get('actual_delta') for e in sol_events]==[0,0,0,50] and all(e.get('expected')==[0,0,0,50] for e in sol_events)
    return {'request_id':request_id,'feature_proof':proof,'selected_backend':'sage_explicit_qk8_pv16' if config.get('stage4_arm',config['candidate_id'])=='adaln_kitchen_sage' else 'fa','observed_backends':observed,'fallback_detected':not backend_ok,'perf_report_path':str(perf_path),'perf_report':report,'perf_report_error':None,'request_identity_match':report.get('request_id')==request_id,'forward_count':len(steps) if valid else 0,'forward_records':steps,'status':'passed' if valid and proof['passed'] and backend_ok else 'blocked_missing_native_evidence'}

def main():
    if not __debug__:raise RuntimeError('optimized Python is forbidden')
    arm=sys.argv[1];purpose=sys.argv[2];output=Path(sys.argv[3])
    assert arm in ('control','adaln_kitchen','adaln_kitchen_sage','adaln_kitchen_sol')
    assert purpose in ('speed','quality')
    assert not output.exists(),'output exists'
    compose=load_composer()
    config=compose.compose_stage4_config(rf,native,arm,purpose)
    if purpose=='quality':
        sliced=records16(native,os.environ.get('STAGE4_PROMPT_SLICE','0:16'))('quality')
        native._prompt_records=records16(native,os.environ.get('STAGE4_PROMPT_SLICE','0:16'))
        config['prompt_records']=sliced
        assert len(config['prompt_records'])==len(sliced)>=1 and all(r['purpose']=='quality' for r in config['prompt_records'])
        config['warmup_probe']=True
        native._dispatch_evidence=quality_dispatch
    config['stage4_wave']={'authority':'owner 2026-09-17 stage4 instruction','arm':arm,'purpose':purpose,'launched_epoch':time.time(),'pairing':'within-hardware per-prompt' if purpose=='quality' else 'n1 descriptive speed'}
    (output.parent/(output.name+'.wave-config.json')).write_text(json.dumps({'method':arm,'purpose':purpose,'candidate_id':config['candidate_id'],'sampler':'euler','env':{k:v for k,v in config['env'].items() if k.startswith(('MINIACC','SGLANG'))},'argv_quantization':config.get('quantization'),'adaln':config.get('adaln_sidecar') is not None or 'adaln' in json.dumps(config.get('argv',[])),'stage4_wave':config['stage4_wave']},indent=2)+'\n')
    return native.execute_request_campaign(config,output,timeout_seconds=7200)
if __name__=='__main__':raise SystemExit(main())
