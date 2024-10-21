REM @echo off
setlocal enabledelayedexpansion

set llm_names=microsoft/Phi-3-mini-4k-instruct meta-llama/Llama-3.2-3B-Instruct
set loss_names=BCE WEIGHTED_BCE CORRECT_AWARE WEIGHTED_CORRECT_AWARE
set dset_names=GSMCoT MMLUCoT AQUARATCoT

for %%m in (%llm_names%) do (
  for %%d in (%dset_names%) do (
    for %%l in (%loss_names%) do (
      python compile_results.py --model_name=%%m --ood_input_formatter_name=%%d --loss_func_name=%%l
    )
  )
)
