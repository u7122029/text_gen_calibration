from datasets import load_dataset, concatenate_datasets

from .dictdataset import DictDataset


def get_aqua_rat():
    def func(questions, options):
        out_dict = {"question": []}
        for question, option in zip(questions, options):
            out_dict["question"].append(f"{question}\n{'\n'.join(option)}")

        return out_dict

    val_dataset = load_dataset("deepmind/aqua_rat", split="validation")
    test_dataset = load_dataset("deepmind/aqua_rat", split="test")
    dataset = concatenate_datasets([val_dataset, test_dataset])
    dataset = dataset.map(func, batched=True, input_columns=["question", "options"], remove_columns=["rationale", "options"])
    dataset = dataset.rename_column("correct", "answer")
    return DictDataset(dataset.to_dict())


if __name__ == "__main__":
    x = get_aqua_rat()
    print(x.keys())
    print(x["answer"])