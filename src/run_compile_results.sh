for dset_name in GSMCoT MMLUCoT AQUARATCoT
do
  for loss_name in BCE WEIGHTED_BCE CORRECT_AWARE WEIGHTED_CORRECT_AWARE
  do
    python compile_results.py --ood_input_formatter_name=$dset_name --loss_func_name=$loss_name;
  done
done

