import fire
import torch

from calibrators import calibrator_dict
from data import DictDataset
from input_formatters import input_formatter_dict, InputFormatter
from llm_models.textgen import TextGenLLMBundle

from metrics import ModelMetrics
from prompt_formatters import PromptVersion
from utils import LossFunc


def show_results(val_data: DictDataset,
                 test_data: DictDataset,
                 model_name: str,
                 calibrator_name: str,
                 input_formatter_name: str,
                 loss_fn_name: str):
    details = {"LLM": model_name,
               "Calibrator": calibrator_name,
               "Input Formatter": input_formatter_name,
               "Loss Function": loss_fn_name}
    val_results = ModelMetrics(val_data, **details)
    test_results = ModelMetrics(test_data, **details)
    print("---")
    print("### Validation Set Results")
    val_results.display()
    print("---")
    print("### Test Set Results")
    test_results.display()


def main(input_formatter_name: str="SQUADV2CoT",
         calibrator_name="APRICOT_Original",
         loss_fn="BCE",
         cot_version="DEFAULT",
         model_name="google/gemma-2-2b-it",
         batch_size=4,
         calib_dset_size=None,
         test_dset_size=None,
         recompute_logits=False,
         retrain_calibrator=False):
    torch.manual_seed(0)
    if calibrator_name not in calibrator_dict:
        raise ValueError(f"calibrator_name '{calibrator_name}' not in {calibrator_dict.keys()}")

    llm_bundle = TextGenLLMBundle(model_name)
    loss_func = LossFunc.from_string(loss_fn)

    cot_version = PromptVersion.from_string(cot_version)
    calibrator_type = calibrator_dict[calibrator_name]
    input_formatter_class = input_formatter_dict[input_formatter_name]
    input_formatter: InputFormatter = input_formatter_class(llm_bundle,
                                                            cot_version,
                                                            calibrator_type,
                                                            loss_func,
                                                            calib_dset_size,
                                                            test_dset_size)

    val_data, test_data = input_formatter.run_pipeline(batch_size,
                                                       recompute_logits=recompute_logits,
                                                       recalibrate=retrain_calibrator)

    show_results(val_data, test_data, model_name, calibrator_name, input_formatter_name, loss_fn)


if __name__ == "__main__":
    fire.Fire(main)
