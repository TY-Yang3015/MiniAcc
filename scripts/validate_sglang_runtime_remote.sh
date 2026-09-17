#!/usr/bin/env bash
# Non-generative SGLang/H3 runtime gate. Run on wolf8 after snapshot download.
set -Eeuo pipefail
project=${MINIACC_PROJECT_ROOT:-$PWD}
python=${MINIACC_SGLANG_PYTHON:-$project/.local/sglang-venv/bin/python}
cli=${MINIACC_SGLANG_CLI:-$project/.local/sglang-venv/bin/sglang}
model=${MINIACC_SGLANG_MODEL:-$(find "$project/.local/sglang-hf-cache" -path '*/snapshots/*' -type d -print -quit 2>/dev/null)}
root=${MINIACC_VALIDATION_ROOT:-$project/artifacts/stage2-sglang-h3-setup}
log=$root/runtime-validation.log
mkdir -p "$root"
[[ -x "$python" && -x "$cli" && -n "$model" ]] || { echo 'missing SGLang executable/CLI or completed model snapshot' >&2; exit 2; }
[[ -f "$model/FL2VA/model_index.json" && -f "$model/FL2VA/transformer/config.json" && -f "$model/FL2VA/text_encoder/config.json" && -f "$model/FL2VA/video_vae/config.json" && -f "$model/FL2VA/audio_vae/config.json" ]]
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PATH="${MINIACC_FFMPEG_DIR:-$project/.local/ffmpeg/bin}:$PATH"
cuda_lib=$("$python" -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13/lib
cuda_home=$project/.local/sglang-cuda
[[ -f "$cuda_lib/libcudart.so.13" ]] || { echo "missing isolated CUDA runtime: $cuda_lib/libcudart.so.13" >&2; exit 2; }
mkdir -p "$project/.local/sglang-cuda/bin" "$project/.local/sglang-cuda/lib64"
ln -sfn "$cuda_lib/libcudart.so.13" "$project/.local/sglang-cuda/lib64/libcudart.so"
ln -sfn /home/di49map/miniconda3/targets/x86_64-linux/include "$project/.local/sglang-cuda/include"
bash "$project/scripts/install_sglang_cuda_wrapper.sh" "$project" "$cuda_home"
export CUDA_HOME="$cuda_home"
export CUDA_PATH="$cuda_home"
export LD_LIBRARY_PATH="$project/.local/sglang-cuda/lib64:$cuda_lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$project/.local/sglang-cuda/lib64:$cuda_lib:${LIBRARY_PATH:-}"
host=$(hostname); start=$(date -Is); printf '{"host":"%s","pid":%s,"start":"%s","status":"running"}\n' "$host" "$$" "$start" > "$root/runtime-validation.start.json"
printf 'validation host=%s pid=%s start=%s model=%s\n' "$host" "$$" "$start" "$model" > "$log"
finish() { rc=$?; now=$(date -Is); printf '{"host":"%s","pid":%s,"start":"%s","exit":"%s","returncode":%s}\n' "$host" "$$" "$start" "$now" "$rc" > "$root/runtime-validation.exit.json"; exit "$rc"; }
trap finish EXIT
stage() {
  name=$1; seconds=$2; shift 2
  printf 'BEFORE stage=%s deadline=%ss at=%s\n' "$name" "$seconds" "$(date -Is)" | tee -a "$log"
  if timeout --signal=TERM --kill-after=10 "$seconds" "$@" >> "$log" 2>&1; then
    printf 'AFTER stage=%s result=passed at=%s\n' "$name" "$(date -Is)" | tee -a "$log"
  else
    rc=$?; printf 'AFTER stage=%s result=failed returncode=%s at=%s\n' "$name" "$rc" "$(date -Is)" | tee -a "$log"; return "$rc"
  fi
}
stage cpu_sglang_import 90 "$python" -u - <<'PY'
import json, sys, sglang
print(json.dumps({"sglang": getattr(sglang, "__version__", "unknown"), "file": sglang.__file__}), flush=True)
PY
stage cuda_init 90 "$python" -u - <<'PY'
import json, torch
print(json.dumps({"torch":torch.__version__,"torch_cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available()},sort_keys=True), flush=True)
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
print(json.dumps({"device":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0),"free_total":torch.cuda.mem_get_info(0)},sort_keys=True), flush=True)
torch.cuda.synchronize()
PY
stage sglang_fa_kernel_smoke 180 "$python" -u - <<'PY'
import json, torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
print(json.dumps({"kernel":"sglang.kernels.ops.attention.flash_attention.flash_attn_varlen_func"}), flush=True)
q=torch.randn((128,4,128),device="cuda",dtype=torch.bfloat16)
cu=torch.tensor([0,128],device="cuda",dtype=torch.int32)
y=flash_attn_varlen_func(q,q,q,cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=128,max_seqlen_k=128,causal=False)
torch.cuda.synchronize()
print(json.dumps({"sglang_fa_sm80_smoke":"passed","shape":list(y.shape)},sort_keys=True), flush=True)
PY
stage sglang_qknorm_jit_smoke 240 "$python" -u - <<'PY'
import json, torch
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
q=torch.randn((128,4,128),device="cuda",dtype=torch.bfloat16)
k=torch.randn_like(q)
qw=torch.ones((128,),device="cuda",dtype=torch.bfloat16)
kw=torch.ones((128,),device="cuda",dtype=torch.bfloat16)
fused_inplace_qknorm(q,k,qw,kw,head_dim=128)
torch.cuda.synchronize()
print(json.dumps({"sglang_qknorm_jit_sm80_smoke":"passed","shape":list(q.shape)}), flush=True)
PY
stage h3_asset_check 30 "$python" -u - "$model" <<'PY'
import pathlib, sys
m=pathlib.Path(sys.argv[1])
for rel in ("FL2VA/model_index.json","FL2VA/transformer/config.json","FL2VA/text_encoder/config.json","FL2VA/video_vae/config.json","FL2VA/audio_vae/config.json"):
 p=m/rel
 print(rel, p.is_file(), p.stat().st_size if p.is_file() else 0, flush=True)
 if not p.is_file(): raise SystemExit(f"missing {rel}")
PY
stage h3_cli_help 120 bash -c '"$1" generate --help > "$2/generate-help.txt" 2>&1 && "$1" serve --model-type diffusion --help > "$2/serve-help.txt" 2>&1 && grep -q -- "--model-path" "$2/generate-help.txt" && grep -q -- "--attention-backend" "$2/serve-help.txt" && grep -q -- "--lora-path" "$2/serve-help.txt" && grep -q -- "--warmup-num-frames" "$2/serve-help.txt"' _ "$cli" "$root"
echo runtime_validation=passed | tee -a "$log"
