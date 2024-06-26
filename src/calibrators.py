from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset
from icecream import ic
import inspect, sys
from nltk import corpus
import pandas as pd
import numpy as np
from torch import nn, optim
from utils import TokenLogitsDataset
import warnings
from torch.nn.functional import relu


class Calibrator(ABC):
    """
    To use a Calibrator, you must
    1. Run the calibrate method to generate a new instance of the given calibrator, or
       load the calibrator llm weights to generate the calibrator llm.
    2. Run the test method to test the calibrator on some data.

    You may use the save method to save the parts of the calibrator that require persistence.
    Note that the save method may not do anything at all if there is nothing to save.
    """
    def __init__(self, tokeniser, model, debug_responses):
        self.tokeniser = tokeniser
        self.model = model
        self.debug_responses = debug_responses
        self.calibrator_model = None

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def calibrate(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class LogitTokenToConfidenceCalibrator(Calibrator):
    def __init__(self, tokeniser, model, calibrator_model, debug_responses=False):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = calibrator_model
        self.tuned = False

    def calibrate(self, calib_tokens, calib_logits, correct, batch_size, **kwargs):
        # Assume calib_logits has shape [dset_length, response_length, vocab_size]
        calibration_dset = TokenLogitsDataset(calib_logits, calib_tokens, correct)
        calibration_dl = DataLoader(calibration_dset,
                                    batch_size=batch_size,
                                    collate_fn=TokenLogitsDataset.collate_fn)
        # Optimise llm.
        self.calibrator_model = self.calibrator_model.cuda()
        loss_fn = nn.MSELoss().cuda() # calibration aware loss with l2 norm squared.
        optimiser = optim.SGD(self.calibrator_model.parameters(), lr=0.01)

        print("Training Calibrator")
        self.calibrator_model.train()
        total_loss_last_epoch = None
        epochs = 30
        for epoch_idx in range(epochs):
            pbar = tqdm(calibration_dl,
                        desc=f"Epoch {epoch_idx + 1}/{epochs}",
                        postfix={"total_loss_last_epoch": total_loss_last_epoch})
            total_loss_last_epoch = 0
            for logits_batch, _, is_correct_batch in pbar:
                logits_batch = logits_batch.to("cuda")
                is_correct_batch = is_correct_batch.to("cuda")

                optimiser.zero_grad()
                out_token_confs = self.calibrator_model(logits_batch)
                loss = loss_fn(out_token_confs, is_correct_batch)
                loss.backward()
                optimiser.step()
                total_loss_last_epoch += loss.item()
        self.calibrator_model.eval()
        self.tuned = True

    def load(self, filepath):
        self.calibrator_model = TemperatureScalingVariant.TSModel()
        self.calibrator_model.load_state_dict(torch.load(filepath))
        self.calibrator_model.eval()

    def save(self, filepath):
        torch.save(self.calibrator_model.state_dict(), filepath)

    def test(self, test_tokens, test_logits, correct, **kwargs):
        if not self.tuned:
            warnings.warn("Calibrator model has not been loaded or trained. Expect dubious results.")
        test_dset = TokenLogitsDataset(test_logits, test_tokens, correct)

        # Get results.
        print("Getting Results")
        confs_after_calibration = []
        with torch.no_grad():
            for logits, tokens, _ in test_dset:
                logits = logits.cuda()
                tokens = tokens.cuda()
                token_confs = self.calibrator_model(logits, tokens).cpu()
                out = torch.mean(token_confs)
                confs_after_calibration.append(out)
        return torch.Tensor(confs_after_calibration)


class TemperatureWithLinearity(LogitTokenToConfidenceCalibrator):
    """
    Uses a model that contains as many temperature parameters as the vocabulary size.
    Idea is that the calibrator should not affect the generation results. So it's model -> base outputs -> calibrator -> confidence.
    Each logit vector is a probability distribution. So in reality, any token can be picked.
    """
    class LTModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.linear_weight = nn.Parameter(torch.ones(1))
            self.linear_bias = nn.Parameter(torch.zeros(self.vocab_size))

        def forward(self, x, tokens=None):
            x = x / self.linear_weight + self.linear_bias # elementwise multiplication.
            x = torch.softmax(x, dim=1)
            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values
            return x  # [confs]

    def __init__(self, tokeniser, model, debug_responses=False):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = TemperatureWithLinearity.LTModel(tokeniser.vocab_size)


class LinearScaler(LogitTokenToConfidenceCalibrator):
    """
    Uses a model that contains as many temperature parameters as the vocabulary size.
    Idea is that the calibrator should not affect the generation results. So it's model -> base outputs -> calibrator -> confidence.
    Each logit vector is a probability distribution. So in reality, any token can be picked.
    """
    class LinearModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
            self.linear_weights = nn.Parameter(torch.ones(self.vocab_size))
            self.linear_bias = nn.Parameter(torch.zeros(self.vocab_size))

        def forward(self, x, tokens=None):
            x = self.linear_weights * x + self.linear_bias # elementwise multiplication.
            x = torch.softmax(x, dim=1)
            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values
            return x  # [confs]

    def __init__(self, tokeniser, model, debug_responses=False):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = LinearScaler.LinearModel(tokeniser.vocab_size)


class TemperatureScalingVariant(LogitTokenToConfidenceCalibrator):
    class TSModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, x, tokens=None):
            # x.shape: [logit_vec, vocab size]
            x = x / self.temperature
            x = torch.softmax(x, dim=1)
            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values
            return x  # [confs]

    def __init__(self, tokeniser, model, debug_responses=False):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = TemperatureScalingVariant.TSModel()


class PTSVariant(LogitTokenToConfidenceCalibrator):
    class PTSModel(nn.Module):
        def __init__(
                self,
                length_logits,
                nlayers=2,
                n_nodes=75,
                top_k_logits=150
        ):
            """
            Args:
                length_logits (int): length of logits vector
                epochs (int): number of epochs for PTS model tuning
                lr (float): learning rate of PTS model
                weight_decay (float): lambda for weight decay in loss function
                batch_size (int): batch_size for tuning
                n_layers (int): number of layers of PTS model
                n_nodes (int): number of nodes of each hidden layer
                top_k_logits (int): top k logits used for tuning
            """
            super().__init__()
            assert nlayers >= 0

            self.nlayers = nlayers
            self.n_nodes = n_nodes
            self.length_logits = length_logits
            self.top_k_logits = top_k_logits

            # Build model
            self.layers = []
            if self.nlayers == 0:
                self.layers.append(nn.Linear(in_features=self.top_k_logits, out_features=1))
            else:
                self.layers.append(nn.Linear(in_features=self.top_k_logits, out_features=self.n_nodes))

            for _ in range(self.nlayers - 1):
                self.layers.append(nn.Linear(in_features=self.n_nodes, out_features=self.n_nodes))

            if self.nlayers > 0:
                self.layers.append(nn.Linear(in_features=self.n_nodes, out_features=1))

            self.layers = nn.ModuleList(self.layers)

        def forward(self, inp, tokens=None):
            t, _ = torch.sort(torch.topk(inp, self.top_k_logits, dim=1).values, dim=1, descending=True)
            t = self.layers[0](t)
            if len(self.layers) > 0:
                t = relu(t)

            for layer_idx in range(1, len(self.layers) - 1):
                t = self.layers[layer_idx](t)
                t = relu(t)

            if len(self.layers) > 0:
                t = self.layers[-1](t)

            t = nn.functional.softplus(t)

            x = inp / t
            x = torch.softmax(x, dim=1)

            if tokens is not None:
                x = torch.take_along_dim(x, tokens.unsqueeze(1), dim=1).squeeze(1)
            else:
                x = torch.max(x, dim=1).values

            return x

    def __init__(self, tokeniser, model, debug_responses=False):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = PTSVariant.PTSModel(tokeniser.vocab_size)


class LinearTemperatureScaling(PTSVariant):
    def __init__(self, tokeniser, model, debug_responses=False):
        super().__init__(tokeniser, model, debug_responses)
        self.calibrator_model = LinearTemperatureScaling.PTSModel(tokeniser.vocab_size, nlayers=1, top_k_logits=tokeniser.vocab_size)


class StopwordRemover(Calibrator):
    def test(self, **kwargs):
        pass

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
                    indices = torch.where(torch.all(response_unfolded == stopword_token_set, dim=1))[0]
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

        all_preds = torch.cat(all_preds)
        confs_after_calib = torch.Tensor(all_calibrated_confs)
        confs_before_calib = torch.Tensor(all_uncalibrated_confs)
        return all_preds, confs_before_calib, confs_after_calib, None


class TopKTokenPoolingV1(Calibrator):
    def test(self, **kwargs):
        pass

    class TokenPoolingCalibrator:
        def __init__(self, tokens_to_filter: torch.Tensor, eos_token: int, temp_tokeniser=None):
            self.tokens_to_filter = torch.ones(len(tokens_to_filter) + 1).int()
            self.tokens_to_filter[:-1] = tokens_to_filter
            self.tokens_to_filter[-1] = eos_token
            self.temp_tokeniser = temp_tokeniser
            print(self.temp_tokeniser.batch_decode(self.tokens_to_filter))
            print("---")

        def __call__(self, token_responses, token_confidences):
            outs = []
            for tokens, confidences in zip(token_responses, token_confidences):
                mask = torch.BoolTensor([True] * len(tokens))
                for token_to_filter in self.tokens_to_filter:
                    mask = mask & (tokens != token_to_filter)
                if torch.all(~mask).item():
                    outs.append(0)
                    continue
                filtered_confidences = confidences[mask]
                outs.append(torch.mean(filtered_confidences).item())

            return torch.Tensor(outs)

    def __init__(self, tokeniser, model, debug_responses):
        super().__init__(tokeniser, model, debug_responses)

    def calibrate(self, dataloader: DataLoader, formatter_cls, k=20, **kwargs):
        incorrect_pooled = {}  # token_idx -> list of confidences
        incorrect_token_responses = {}  # token_idx -> list of response numbers

        all_token_confidences = []
        all_token_responses = []
        confs_before_calibration = []
        confs_after_calibration = []
        all_preds = []

        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            formatted = batch["formatted"]
            answers = batch["answer"]

            inputs = self.tokeniser(formatted, return_tensors="pt", padding=True).to("cuda")
            generated = self.model.generate(**inputs,
                                            max_new_tokens=550,
                                            output_logits=True,
                                            return_dict_in_generate=True,
                                            pad_token_id=self.tokeniser.eos_token_id)

            out_dict = formatter_cls.process_responses(inputs, generated, self.tokeniser, get_confidence=True)
            final_answers = out_dict["final_answers"]
            token_confidences = out_dict["token_confidences"]
            correct = answers == final_answers
            all_preds.append(final_answers)

            start = batch_idx * dataloader.batch_size
            end = start + dataloader.batch_size

            # iterate through responses to categorise every outputted token.
            for response_idx, is_correct, token_confs, tokens in zip(range(start, end),
                                                                     correct,
                                                                     token_confidences,
                                                                     out_dict["tokens"]):
                # Remove the eos tokens.
                eos_mask = torch.ones(len(tokens))  # 0 for <eos>, 1 otherwise
                eos_indices = torch.where(tokens == self.tokeniser.eos_token_id)[0]
                eos_mask[eos_indices] = 0
                eos_mask = eos_mask.bool()

                token_confs = token_confs[eos_mask]
                tokens = tokens[eos_mask]
                all_token_confidences.append(token_confs)
                all_token_responses.append(tokens)
                confs_before_calibration.append(torch.mean(token_confs).item())

                if is_correct.item():
                    continue

                token_responses_dict = incorrect_token_responses

                for token, token_conf in zip(tokens, token_confs):
                    token = token.item()
                    token_conf = token_conf.item()
                    if token not in incorrect_pooled:
                        incorrect_pooled[token] = []
                    if token not in token_responses_dict:
                        token_responses_dict[token] = set()

                    token_responses_dict[token].add(response_idx)
                    incorrect_pooled[token].append(token_conf)
        incorrect_d = {
            "tokens": [],
            "means": [],
            "stds": [],
            "total_occurrences": [],
            "no_responses_appeared": []
        }

        incorrect_pooled_keys = list(incorrect_pooled.keys())
        incorrect_d["tokens_str"] = self.tokeniser.batch_decode(incorrect_pooled_keys, skip_special_tokens=True)
        for i in incorrect_pooled_keys:
            v = incorrect_pooled[i]
            v_tensor = torch.Tensor(v)
            s = incorrect_token_responses[i]

            incorrect_d["tokens"].append(i)
            incorrect_d["total_occurrences"].append(len(v_tensor))
            incorrect_d["no_responses_appeared"].append(len(s))

            std, mean = torch.std_mean(v_tensor, unbiased=False)
            incorrect_d["means"].append(mean.item())
            incorrect_d["stds"].append(std.item())

        incorrect_df = pd.DataFrame.from_dict(incorrect_d)

        incorrect_df["scores"] = (((incorrect_df["means"] / (1 + incorrect_df["stds"]))
                                   * np.log(incorrect_df["total_occurrences"]))
                                  * np.sqrt(incorrect_df["no_responses_appeared"]))

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        incorrect_df = incorrect_df.sort_values("scores", ascending=False)

        top_k_tokens = torch.Tensor(incorrect_df["tokens"].to_numpy())[:k]

        calibrator_model = TopKTokenPoolingV1.TokenPoolingCalibrator(top_k_tokens, self.tokeniser.eos_token_id,
                                                                   self.tokeniser)
        confs_after_calibration = calibrator_model(all_token_responses, all_token_confidences)
        confs_before_calibration = torch.Tensor(confs_before_calibration)
        all_preds = torch.cat(all_preds)
        return all_preds, confs_before_calibration, confs_after_calibration, calibrator_model


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
                factors = 2 ** factors
                conf = torch.sum(torch.Tensor([x[0] for x in d_tuples]) * factors).item()
                answer = d_tuples[0][1]

                all_confs.append(conf)
                all_answers.append(answer)

            #all_explanations.extend(outs["explanations"])
            #compiled_answers.extend(outs["final_answers"])
            #compiled_confs.append(outs["confidences"])
        return all_answers, all_confs


class WATCCalibrator(Calibrator):
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
            truncated_tokens = outs["tokens"]  # List[torch tokens as ints]

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
            inp_next = inp * (-(1 - self.f) / (1 - self.t) * nn.functional.relu(inp - self.t) + 1)
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
calibrator_dict: dict[str, Calibrator.__class__] = {x: y for x, y in classes}

if __name__ == "__main__":
    class FakeTokeniser:
        def __init__(self):
            self.vocab_size = 100
    x = TemperatureWithLinearity(FakeTokeniser(), None)
    print(x.calibrator_model.state_dict())