# How can Token Occurences Calibrate Large Language Model Confidence?

## Abstract
Model calibration involves adjusting the confidence of responses such that they better represent the accuracy of the model in question. That is, if a model has a mean confidence of $$c\%$$ over a set of outputs, then its accuracy should be $$c\%$$. We propose a new group of model calibration heuristics that help adjust the confidences of tokens that appear often between responses, or commonly have high, consistent confidence. These calibration methods can be employed to extend existing calibrators to further adjust the confidences of targeted tokens.  By calibrating a model's confidences in this manner, it would be less overconfident in its incorrect answers, and less underconfident in its correct answers. Although the current state-of-the-art calibratiors show varying levels of success on many text generation Large Language Models (LLMs), they can be further improved by considering intuitive natural language patterns. Our results show that our heuristics can improve the Expected Calibration Error (ECE) of existing calibrators, both on data in the same domain or otherwise, though the calibrators we have tested could have been too small or large to display more consistent improvements. Our work also highlights the complexity of distinguishing the correctness of responses, how larger amounts of training data can aid with calibration and the effects of out-of-distribution data on calibrator models.

## Overview
LLMs often indicate a high level of confidence, regardless of whether the answer provided to a given question is correct or not. This confidence comes in the form of wording,
token-based confidences or obtaining the confidence from the LLM itself. LLMs that cannot give confidences that align with their accuracies tend to give misinformation that appears legitimate.

Though there are many ways to obtain confidence from an LLM, the way we choose to do so is via the mean token confidence. Our proposed methods hence aim to adjust the confidences given by each token.

We propose a new $$\xi$$ metric that assigns a score to each token in the LLM's vocabulary, giving higher scores to those that are more viable for stronger calibration strategies. This metric can take on any form, and we are unsure which one would theoretically perform best, so we devise 5 different versions.
1. $$\xi_{m}(t, R) = \mu(t, R)$$,
2. $$\xi_{s}(t, R) = -2\sigma(t, R) + 1$$, 
3. $$\xi_{r}(t, R) = \text{rfr}(t, R)$$,
4. $$\xi_{mr}(t, r) = (\mu \cdot \text{rfr})(t, R)$$,
5. $$\xi_{sr}(t, r) = (\sigma \cdot \text{rfr})(t, R)$$

Where 
- $$R$$ is the set of responses, 
- $$t$$ is a token, 
- $$\mu$$ is the mean token confidence,
- $$\sigma$$ is the standard deviation of token confidences,
- $$\text{rfr}$$ is the **Response Frequency Ratio** - the proportion of responses that token $$t$$ appears across all responses in $$R$$.

Metric 2 gives higher scores to tokens that have consistent confidences, metric 4 gives higher scores to tokens that have high mean confidence and high response frequency ratio, and so on.

We find that tokens that tend to score high $$\xi$$-scores tend to strongly affect response confidence - we saw this by setting the confidences of these tokens to 0, to which we observed decreases in overall response confidence. 

Our strategy from here was to choose simple calibrators and add some extra parameters that solely focus on adjusting these tokens. In our paper we call these $$\xi$$-calibrators,
while our code prefixes the calibrators with the word `Frequency` or `F` if the name was too long. `complie_results.py` contains the full list of calibrators that we tested.

## Reproducing our Results
1. Ensure you are in the `src` directory. 
    ```
    cd src
    ```
2. Install all necessary packages:
    ```
    pip install -r requirements.txt
    ```
3. If you are on Windows, run 
    ```
    ./run_compile_results.bat
    ```
    If you are on a UNIX system, run
    ```
    bash run_compile_results.sh
    ```

