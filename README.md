# text_gen_calibration
Calibrating Text Generation Large Language Models
# FrequencyTS Variant Results
## google/gemma-1.1-2b-it
<details>
  <summary>View Results</summary>

### Calibration Set
```
ece_logits: 0.881216
ece_verbalised: 0.256036

brier_logits: 0.814754
brier_verbalised: 0.172123

auroc_logits: 0.631076
auroc_verbalised: 0.491197

auprc_logits: 0.0741467
auprc_verbalised: 0.0762646
```
| name                   |   ece_calib |   brier_calib |   auroc_calib |   auprc_calib |
|:-----------------------|------------:|--------------:|--------------:|--------------:|
| FrequencyTS            |  0.00963191 |     0.0383401 |      0.630787 |     0.0652092 |
| FrequencyTSBotOnly     |  0.0230489  |     0.0388787 |      0.591146 |     0.070289  |
| FrequencyTSMeanOnly    |  0.00963191 |     0.0383401 |      0.630787 |     0.0652092 |
| FrequencyTSMeanStdOnly |  0.00963191 |     0.0383401 |      0.630787 |     0.0652092 |
| FrequencyTSNoRF        |  0.00963191 |     0.0383401 |      0.630787 |     0.0652092 |
| FrequencyTSTopOnly     |  0.0106741  |     0.0384153 |      0.618056 |     0.0634603 |

### Test Set
```
ece_logits: 0.853072
ece_verbalised: 0.254713

brier_logits: 0.789192
brier_verbalised: 0.169274

auroc_logits: 0.630536
auroc_verbalised: 0.542066

auprc_logits: 0.13184
auprc_verbalised: 0.114857
```
| name                   |   ece_calib |   brier_calib |   auroc_calib |   auprc_calib |
|:-----------------------|------------:|--------------:|--------------:|--------------:|
| FrequencyTS            |   0.0244503 |     0.0626174 |      0.544643 |     0.0777686 |
| FrequencyTSBotOnly     |   0.0032739 |     0.0624013 |      0.530536 |     0.0837853 |
| FrequencyTSMeanOnly    |   0.0244503 |     0.0626174 |      0.544643 |     0.0777686 |
| FrequencyTSMeanStdOnly |   0.0244503 |     0.0626174 |      0.544643 |     0.0777686 |
| FrequencyTSNoRF        |   0.0244503 |     0.0626174 |      0.544643 |     0.0777686 |
| FrequencyTSTopOnly     |   0.0164172 |     0.0626752 |      0.5275   |     0.0748827 |

</details>