import torch
from datasets import load_dataset

from data.dictdataset import DictDataset


def get_squad_v2():
    """
    Obtains the SQUADv2 validation dataset. We filter out questions that do not have an answer because we are completely
    focused on whether the given answer is correct or not. It also makes the dataset size smaller and easier to run
    inference on.
    @return:
    """
    dataset = DictDataset(load_dataset("rajpurkar/squad_v2",
                                       split="validation",
                                       trust_remote_code=True).to_dict())

    dataset["answer"] = dataset["answers"]
    del dataset["answers"]
    extraction_indices = []

    del dataset["title"]
    #lengths = []
    for i, (context, answer) in enumerate(zip(dataset["context"], dataset["answer"])):
        if len(answer["text"]) == 0:
            continue

        extraction_indices.append(i)
        #lengths.append(len(context.split()))

    #print(f"Average context length: {torch.Tensor(lengths).mean().item()}")
    dataset = dataset[extraction_indices]
    return dataset