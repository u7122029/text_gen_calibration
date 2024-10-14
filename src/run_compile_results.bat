REM @echo off
setlocal enabledelayedexpansion

set loss_names=BCE WEIGHTED_BCE CORRECT_AWARE WEIGHTED_CORRECT_AWARE
set dset_names=GSMCoT MMLUCoT AQUARATCoT

for %%d in (%dset_names%) do (
  for %%l in (%loss_names%) do (
    python compile_results.py --ood_input_formatter_name=%%d --loss_func_name=%%l
  )
)
