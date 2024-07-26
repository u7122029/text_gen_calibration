import fire
from utils import TextGenLLMBundle, RESULTS_PATH
from input_formatters import input_formatter_dict
from calibrators import FrequencyTS
from pathlib import Path
import pandas as pd
from data import get_gsm, DictDataset


def main(model_name: str="google/gemma-1.1-2b-it", input_formatter_name="GSMCoT"):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 3000)

    llm_bundle = TextGenLLMBundle(model_name)
    calibrator = FrequencyTS(llm_bundle)
    path = Path(RESULTS_PATH) / model_name / input_formatter_name
    calib_data = DictDataset.from_file(path / "calibration_data.dill")
    calibrator.compute_scores_and_indices(calib_data)
    print(calibrator.compile_token_score().head(500))


if __name__ == "__main__":
    fire.Fire(main)