import fire
from input_formatters import input_formatter_dict, InputFormatter, CoTInputFormatter
from calibrators import calibrator_dict
from llm_models import TextGenLLMBundle
from utils import RESULTS_PATH, dill_load
from pathlib import Path
from data import DictDataset
import pandas as pd
from metrics import ModelMetrics, ModelMetricsCollection


def main(model_name: str="google/gemma-1.1-2b-it", input_formatter_name: str="GSMCoT"):
    input_formatter: InputFormatter = input_formatter_dict[input_formatter_name]
    results_root = Path(RESULTS_PATH)
    metric_results_calib = ModelMetricsCollection()
    metric_results_test = ModelMetricsCollection()

    llm_bundle = TextGenLLMBundle(model_name)
    input_formatter = input_formatter(llm_bundle, 300, 300)
    for calibrator_name in ['FrequencyTS', 'FrequencyTSBotOnly', 'FrequencyTSMeanOnly', 'FrequencyTSMeanStdOnly', 'FrequencyTSNoRF', 'FrequencyTSTopOnly']:
        calibrator = calibrator_dict[calibrator_name]
        calib_data, test_data = input_formatter.run_calibration_pipeline(calibrator)
        results_dir = results_root / model_name / input_formatter.__class__.__name__ / calibrator_name
        calib_results = dill_load(results_dir / "calib_results.dill")
        test_results = dill_load(results_dir / "test_results.dill")
        calib_data.update(calib_results)
        test_results.update(test_results)

        metric_results_calib.append(ModelMetrics(calib_data, calibrator_name))
        metric_results_test.append(ModelMetrics(test_data, calibrator_name))

    #for metric_result in metric_results_calib:
    #    print(len(metric_result))
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(metric_results_calib.generate_tables())
    print(metric_results_test.generate_tables())


if __name__ == "__main__":
    fire.Fire(main)