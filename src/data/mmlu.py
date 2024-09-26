from datasets import load_dataset, concatenate_datasets
from datasets import get_dataset_config_names

from .dictdataset import DictDataset


def get_mmlu():
    def func(questions, choices, answers):
        out_dict = {"question": [], "answer": []}
        choice_names = ["A", "B", "C", "D"]
        for question, choice, answer in zip(questions, choices, answers):
            choice = [f"{name}: {c}" for c, name in zip(choice, choice_names)]
            out_dict["question"].append(f"{question}\n{'\n'.join(choice)}")
            out_dict["answer"].append(choice_names[answer])

        return out_dict

    dset_name = "cais/mmlu"
    datasets = []
    for name in get_dataset_config_names(dset_name):
        if (name.startswith("high") or name.startswith("college")) and not name.endswith("mathematics"):
            print(f"mmlu: loading {name}")
            datasets.append(load_dataset(dset_name, name=name, split="test"))

    dataset = concatenate_datasets(datasets)
    dataset = dataset.rename_column('answer', 'num_answer')
    dataset = dataset.map(func,
                          batched=True,
                          input_columns=["question", "choices", "num_answer"],
                          remove_columns=["subject", "choices", "num_answer"])
    return DictDataset(dataset.to_dict())


if __name__ == "__main__":
    x = get_mmlu()
    #print(x.keys())
    print(x["answer"])
    print(len(x))