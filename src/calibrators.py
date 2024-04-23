from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from icecream import ic
from torch import nn, optim
import inspect, sys
from nltk import corpus


class Calibrator(ABC):
    def __init__(self, tokeniser, model, debug_responses):
        self.tokeniser = tokeniser
        self.model = model
        self.debug_responses = debug_responses

    @abstractmethod
    def calibrate(self, dataloader: DataLoader, formatter_cls, **kwargs):
        pass


class StopwordRemover(Calibrator):
    def __init__(self, tokeniser, model, debug_responses):
        super().__init__(tokeniser, model, debug_responses)
        self.stopwords = corpus.stopwords.words('english')
        self.stopwords += [f" {x.capitalize()}" for x in self.stopwords]
        self.stopwords += [f" {x}" for x in self.stopwords]
        self.stopwords += [x.capitalize() for x in self.stopwords]

        # convert the stopword strings into token indices + their attention masks.
        generated = self.tokeniser(self.stopwords,
                                   return_tensors="pt",
                                   padding=True,
                                   add_special_tokens=False)

        stopword_tokens = generated.input_ids
        stopword_attention_mask = generated.attention_mask

        # Remove the padding.
        self.stopword_tokens = [stopword_token_set[attention_mask.bool()]
                                for stopword_token_set, attention_mask in zip(stopword_tokens, stopword_attention_mask)]

    def calibrate(self, dataloader: DataLoader, formatter_cls, **kwargs):
        all_preds = []
        all_calibrated_confs = []
        all_uncalibrated_confs = []
        all_correct_answers = []

        # First obtain the probabilities and final answers.
        for items in tqdm(dataloader):
            formatted = items["formatted"]
            all_correct_answers.append(items["answer"])
            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")

            generated = self.model.generate(**inputs,
                                            max_new_tokens=550,
                                            output_logits=True,
                                            return_dict_in_generate=True,
                                            pad_token_id=self.tokeniser.eos_token_id
                                            )
            if self.debug_responses:
                ic(self.tokeniser.batch_decode(generated.sequences))

            outs = formatter_cls.process_responses(inputs, generated, self.tokeniser, get_confidence=False)
            all_preds.append(torch.Tensor(outs["final_answers"]))

            prob_vecs = torch.softmax(torch.stack(generated.logits).permute(1, 0, 2), dim=2).cpu()
            sequences = generated.sequences.cpu()
            responses = sequences[:, inputs.input_ids.shape[1]:]
            responses_after_stopword_removal = []

            for response, prob_vec in zip(responses, prob_vecs):
                # Find the eos tokens.
                eos_mask = torch.ones(len(response))
                eos_indices = torch.where(response == self.tokeniser.eos_token_id)[0]
                eos_mask[eos_indices] = 0
                eos_mask = eos_mask.bool()

                # Find the stopword tokens.
                stopword_mask = torch.ones(len(response))
                for stopword_token_set in self.stopword_tokens:
                    response_unfolded = response.unfold(0, len(stopword_token_set), 1)
                    indices = torch.where(torch.all(response_unfolded == stopword_token_set))[0]
                    stopword_mask[indices] = 0
                stopword_mask = stopword_mask.bool()

                uncalibrated_conf = torch.mean(torch.take_along_dim(prob_vec[eos_mask],
                                                                    response[eos_mask].unsqueeze(1), dim=1).squeeze(1))

                modified_response = response[stopword_mask & eos_mask]
                responses_after_stopword_removal.append(modified_response)
                calibrated_conf = torch.mean(torch.take_along_dim(prob_vec[stopword_mask & eos_mask],
                                                                  modified_response.unsqueeze(1), dim=1).squeeze(1))
                all_calibrated_confs.append(calibrated_conf)
                all_uncalibrated_confs.append(uncalibrated_conf)

                ic(self.tokeniser.batch_decode([response, modified_response]))
                quit()

        all_preds = torch.cat(all_preds)
        confs_after_calib = torch.Tensor(all_calibrated_confs)
        confs_before_calib = torch.Tensor(all_uncalibrated_confs)
        return all_preds, confs_before_calib, confs_after_calib, None


class GSDCalibrator(Calibrator):
    def __init__(self, tokeniser, model, debug_responses=True):
        super().__init__(tokeniser, model, debug_responses)

    def calibrate(self, dataloader: DataLoader, formatter_cls, num_trials_per_input=5):
        #all_explanations = []
        all_answers = []
        all_confs = []
        for items in tqdm(dataloader):
            formatted = items["formatted"]
            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")

            compiled_answers = [{} for _ in range(dataloader.batch_size)]
            for trial_idx in range(num_trials_per_input):
                generated = self.model.generate(**inputs,
                                                max_new_tokens=550,
                                                output_logits=True,
                                                return_dict_in_generate=True
                                                )

                outs = formatter_cls.process_responses(inputs, generated, self.tokeniser)
                for idx, final_answer in enumerate(outs["final_answers"]):
                    if final_answer not in compiled_answers[idx]:
                        compiled_answers[idx][final_answer] = 1
                        continue
                    compiled_answers[idx][final_answer] += 1

            #confs_and_answers = [sorted([(v / num_trials_per_input, k) for k, v in d.items()], reverse=True)
            #                     for d in compiled_answers]
            for d in compiled_answers:
                d_tuples = sorted([(v / num_trials_per_input, k) for k, v in d.items()], reverse=True)
                factors = torch.arange(len(d_tuples))
                factors[1:] = -factors[1:]
                factors = 2**factors
                conf = torch.sum(torch.Tensor([x[0] for x in d_tuples]) * factors).item()
                answer = d_tuples[0][1]

                all_confs.append(conf)
                all_answers.append(answer)

            #all_explanations.extend(outs["explanations"])
            #compiled_answers.extend(outs["final_answers"])
            #compiled_confs.append(outs["confidences"])
        return all_answers, all_confs


class WATCCalibrator(Calibrator, ABC):
    def __init__(self, tokeniser, model, calibrator_model, debug_responses):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = calibrator_model

    def calibrate(self, dataloader: DataLoader, formatter_cls, epochs=50):
        all_preds = []
        all_confs = []
        criterion = nn.BCELoss()
        # optimiser = optim.LBFGS(calibrator.parameters(), 1e-4, max_iter=epochs)
        optimiser = optim.Adam(self.calibrator_model.parameters(), 5e-4)

        all_correct_answers = []

        # First obtain the probabilities and final answers.
        for items in tqdm(dataloader):
            formatted = items["formatted"]
            all_correct_answers.append(items["answer"])
            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")

            generated = self.model.generate(**inputs,
                                            max_new_tokens=550,
                                            output_logits=True,
                                            return_dict_in_generate=True,
                                            pad_token_id=self.tokeniser.eos_token_id
                                            )

            outs = formatter_cls.process_responses(inputs, generated, self.tokeniser, get_confidence=False)
            truncated_tokens = outs["tokens"] # List[torch tokens as ints]

            if self.debug_responses:
                ic(outs["final_answers"])

            prob_vecs = torch.softmax(torch.stack(generated.logits).permute(1, 0, 2), dim=2).cpu()
            compiled_probs = []
            for truncated_token_set, prob_vec in zip(truncated_tokens, prob_vecs):
                prob = torch.take_along_dim(prob_vec[:len(truncated_token_set)],
                                            truncated_token_set.unsqueeze(1), dim=1).squeeze(1)
                compiled_probs.append(prob)

            all_preds.append(torch.Tensor(outs["final_answers"]))
            all_confs.append(torch.stack(compiled_probs))
        
        all_preds = torch.cat(all_preds)

        self.calibrator_model.train()
        progress = tqdm(range(epochs), desc="optimising")
        for _ in progress:
            total_loss = 0
            for conf_batch, correct_answers, answers in zip(all_confs, all_correct_answers, all_preds):
                optimiser.zero_grad()
                loss = criterion(self.calibrator_model(conf_batch), (correct_answers == answers).float())
                loss.backward()
                optimiser.step()
                total_loss += loss
                for p in self.calibrator_model.parameters():
                    p.data.clamp_(0, 1.0)
            progress.set_postfix({"total_loss": total_loss.item()})

        self.calibrator_model.eval()
        confs_after_calib = torch.cat([self.calibrator_model(confs) for confs in all_confs])
        confs_before_calib = torch.cat([torch.mean(batch, dim=1) for batch in all_confs])
        return all_preds, confs_before_calib, confs_after_calib, self.calibrator_model


class ReLu_WATC(WATCCalibrator):
    class ReLuCalibrator(nn.Module):
        def __init__(self, f=0.5, t=0.5):
            super().__init__()
            self.f = nn.Parameter(torch.Tensor([f]))
            self.t = nn.Parameter(torch.Tensor([t]))

        def forward(self, inp):
            inp_next = inp *(-(1 - self.f)/(1 - self.t) * nn.functional.relu(inp - self.t) + 1)
            confidences = inp_next.mean(dim=1)
            return confidences

    def __init__(self, tokeniser, model, debug_responses, f=0.5, t=0.5):
        super().__init__(tokeniser, model, ReLu_WATC.ReLuCalibrator(f, t), debug_responses)


class Step_WATC(WATCCalibrator):
    class StepCalibrator(nn.Module):
        def __init__(self, t=0.5, f=0.5):
            super().__init__()
            self.f = nn.Parameter(torch.Tensor([f]))
            self.t = nn.Parameter(torch.Tensor([t]))

        def forward(self, inp):
            mask = inp >= self.t
            inp_next = inp[mask] * self.f
            confidences = inp_next.mean(dim=1)
            return confidences

    def __init__(self, tokeniser, model, debug_responses, t=0.5, f=0.5):
        super().__init__(tokeniser, model, Step_WATC.StepCalibrator(t, f), debug_responses)


def get_class_bases(x):
    bases = set()
    for base in x.__bases__:
        bases.add(base)
        bases = bases.union(get_class_bases(base))
    return bases


def dset_class_predicate(x):
    if not inspect.isclass(x): return False

    class_bases = get_class_bases(x)
    return Calibrator in class_bases


classes = inspect.getmembers(sys.modules[__name__], dset_class_predicate)
calibrator_dict: dict[str, Calibrator] = {x: y for x, y in classes}
