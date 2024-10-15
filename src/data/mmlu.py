from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from datasets import get_dataset_config_names, load_from_disk

from utils.earlystopping import TEMP_DIR
from .dictdataset import DictDataset


def construct_mmlu(out_path: Path):
    print("MMLU dataset not found on disk. Constructing dataset from scratch.")
    dset_name = "cais/mmlu"
    datasets = []
    #subjects = {"machine_learning", "computer_security", "econometrics", "jurisprudence", "philosophy", "prehistory"}
    subjects = set()
    for name in get_dataset_config_names(dset_name):
        if (
                (name.startswith("high") or name.startswith("college"))
                and not name.endswith("mathematics")
        ) or name in subjects:
            print(f"mmlu: loading {name}")
            datasets.append(load_dataset(dset_name, name=name, split="test"))

    dataset = concatenate_datasets(datasets)
    dataset.save_to_disk(str(out_path))
    print("MMLU Dataset saved to disk.")
    return dataset


def get_mmlu():
    def func(questions, choices, answers):
        out_dict = {"question": [], "answer": []}
        choice_names = ["A", "B", "C", "D"]
        for question, choice, answer in zip(questions, choices, answers):
            choice = [f"{name}: {c}" for c, name in zip(choice, choice_names)]
            out_dict["question"].append(f"{question}\n{'\n'.join(choice)}")
            out_dict["answer"].append(choice_names[answer])

        return out_dict

    path = Path(TEMP_DIR) / "data" / "mmlu.arrow"
    if not path.exists():
        dataset = construct_mmlu(path)
    else:
        print("Loading MMLU dataset from disk.")
        dataset = load_from_disk(str(path))

    dataset = dataset.rename_column('answer', 'num_answer')
    dataset = dataset.map(func,
                          batched=True,
                          input_columns=["question", "choices", "num_answer"],
                          remove_columns=["subject", "choices", "num_answer"])
    out = DictDataset(dataset.to_dict())
    print(f"mmlu length: {len(out)}")
    return out


if __name__ == "__main__":
    x = get_mmlu()
    #print(x.keys())
    print(x["answer"])
    print(len(x))
