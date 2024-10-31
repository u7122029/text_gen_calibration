REM @echo off
setlocal enabledelayedexpansion

set llm_names=google/gemma-2-2b-it meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-3B-Instruct microsoft/Phi-3-mini-4k-instruct Zyphra/Zamba2-2.7B-instruct meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct
set loss_names=BCE WEIGHTED_BCE CORRECT_AWARE WEIGHTED_CORRECT_AWARE
set dset_names=GSMCoT MMLUCoT AQUARATCoT

for %%m in (%llm_names%) do (
  for %%d in (%dset_names%) do (
    for %%l in (%loss_names%) do (
      python compile_results.py --model_name=%%m --ood_input_formatter_name=%%d --loss_func_name=%%l
    )
  )
)
