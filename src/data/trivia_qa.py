from datasets import load_dataset

from data.dictdataset import DictDataset


def get_trivia_qa():
    """
    Obtains the TriviaQA validation dataset. Note that we filter out questions that have more than one source so that
    the length of the context is not as long. Furthermore, the size of the whole dataset is over 4000, so we have not
    lost too much meaningful data.
    @return:
    """
    dataset = DictDataset(load_dataset("mandarjoshi/trivia_qa",
                                       "rc.wikipedia",
                                       split="validation",
                                       trust_remote_code=True).to_dict())
    extraction_indices = []
    contexts = []
    del dataset["question_id"]
    del dataset["question_source"]
    del dataset["search_results"]

    dataset["answer"] = [set(x["normalized_aliases"]) for x in dataset["answer"]]
    for i, x in enumerate(dataset["entity_pages"]):
        if len(x["title"]) == 1:
            extraction_indices.append(i)
            contexts.append(x["wiki_context"][0])
    dataset = dataset[extraction_indices]
    dataset["context"] = contexts
    del dataset["entity_pages"]
    return dataset