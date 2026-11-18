# Appendix

Supplementary material for *Interactive Abilities of Visually Grounded Conversational Models in the LLM Era* (Antonelli, Bernardi, Fernández; Phil. Trans. R. Soc. B, 2026).

- [A. Examples of pairs](#a-examples-of-pairs)
- [B. Sensitivity and stratification analyses](#b-sensitivity-and-stratification-analyses)
- [C. Layer-wise probing trajectory](#c-layer-wise-probing-trajectory)
- [D. Multimodal vs. unimodal error breakdown](#d-multimodal-vs-unimodal-error-breakdown)
- [E. Control for baseline in REC task](#e-control-for-baseline-in-rec-task)

---

## A. Examples of pairs

Here are reported examples of different kinds of descriptions used as "Speaker" in the task:

- **Human written**: descriptions originally collected by asking human participants to write down the description to identify the target.
- **Human spoken**: descriptions originally collected by asking human participants to say out loud the description to identify the target, and then transcribed.
- **Model Default**: descriptions originally collected by prompting VLMs to output a description for the target.
- **Model Brief**: descriptions originally collected by prompting VLMs to output a brief description for the target.

The examples in Figure 4 contain all four descriptions. GPT-4o was picked as an example for the VLM category. Seven more models were used in speaker generation. The figure is adapted from the original RefOI work (Ma et al., 2025).

## B. Sensitivity and stratification analyses

This appendix reports additional analyses supporting the behavioural comprehension results in Section 3(a). Specifically, we report: (i) a logistic regression with image-clustered standard errors, comparing description conditions while controlling for the number of distractors; (ii) a sensitivity analysis with a stricter IoU threshold; and (iii) accuracy stratified by visual clutter (number of same-category distractors).

### B(a). Trial-level regression modelling

To test whether listener success differed across description conditions, we fitted a binary logistic regression on trial-level comprehension outcomes (IoU ≥ 0.5) for QWEN2.5-VL-7b. Standard errors were Huber–White cluster-robust, clustered by image identifier, with degrees of freedom adjusted for the number of clusters. The number of distractors (0 to 4) was entered as a continuous covariate. The verbose model condition (Model Default) was the reference category.

Table 4 reports the results on the full dataset, where the dominant predictor of success is the distractor count, and description source effects remain diluted by uncontrolled baseline trials without distractors. Crucially, when fitting the same model exclusively on the subset where pragmatic disambiguation is actively required (≥ 1 distractor), the advantage of human descriptions over Model Default descriptions becomes statistically significant (Spoken: β = 0.320, OR = 1.38, p = 0.006; Written: β = 0.288, OR = 1.33, p = 0.015), while Model Brief descriptions show no significant difference (β = 0.052, p = 0.592). This demonstrates a measurable, though numerically modest, pragmatic sensitivity when the model must resolve genuine referential ambiguity.

**Table 4.** Logistic regression results on trial-level REC success for QWEN2.5-VL-7b, on the full dataset and on the subset of trials with at least one same-category distractor. The number of distractors (0–4) is entered as a continuous covariate. Reference category for intercept: Model Default.

| Parameter | Full β (SE) | Full p | Full OR [95% CI] | ≥1 distractor β (SE) | ≥1 distractor p | ≥1 distractor OR [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Intercept | 2.354 (0.055) | < 0.001 | 10.53 [9.46, 11.72] | 1.795 (0.094) | < 0.001 | 6.02 [5.01, 7.23] |
| Human Spoken | 0.218 (0.102) | 0.033 | 1.24 [1.02, 1.52] | 0.320 (0.117) | 0.006 | 1.38 [1.10, 1.73] |
| Human Written | 0.197 (0.103) | 0.057 | 1.22 [0.99, 1.49] | 0.288 (0.118) | 0.015 | 1.33 [1.06, 1.68] |
| Model Brief | −0.085 (0.071) | 0.231 | 0.92 [0.80, 1.06] | 0.052 (0.098) | 0.592 | 1.05 [0.87, 1.28] |
| Co-occurrence (Distractors) | −0.378 (0.025) | < 0.001 | 0.68 [0.65, 0.72] | −0.183 (0.039) | < 0.001 | 0.83 [0.77, 0.90] |

### B(b). IoU threshold sensitivity

To check that the results do not depend on the standard IoU ≥ 0.50 threshold, we also evaluated localisation success under a stricter criterion (IoU ≥ 0.75), and report the mean IoU as a threshold-free measure.

**Table 5.** Evaluation of QWEN2.5-VL-7b comprehension across IoU thresholds and mean IoU. GLaMM is based on few trials (126–162 per condition), so its values are unstable.

| Speaker | Condition | Trials | Acc (IoU ≥ 0.50) | Acc (IoU ≥ 0.75) | Mean IoU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Human | Spoken | 1,309 | 88.31% | 75.40% | 0.7892 |
| Human | Written | 1,249 | 88.15% | 76.35% | 0.7953 |
| CogVLM | Brief prompt | 1,428 | 88.80% | 76.68% | 0.8028 |
| CogVLM | Default prompt | 1,081 | 87.60% | 75.12% | 0.7908 |
| GLaMM | Brief prompt | 162 | 59.88% | 52.47% | 0.5779 |
| GLaMM | Default prompt | 126 | 46.83% | 44.44% | 0.4818 |
| GPT-4o | Brief prompt | 2,094 | 88.92% | 80.04% | 0.8127 |
| GPT-4o | Default prompt | 2,201 | 89.05% | 80.69% | 0.8107 |
| LLaVA-13B | Brief prompt | 1,263 | 91.84% | 85.35% | 0.8486 |
| LLaVA-13B | Default prompt | 1,695 | 90.91% | 84.07% | 0.8347 |
| LLaVA-34B | Brief prompt | 1,503 | 89.29% | 80.97% | 0.8185 |
| LLaVA-34B | Default prompt | 1,730 | 88.09% | 80.58% | 0.8095 |
| LLaVA-7B | Brief prompt | 1,206 | 87.48% | 80.76% | 0.8118 |
| LLaVA-7B | Default prompt | 1,490 | 91.07% | 83.02% | 0.8332 |
| MiniCPM-V | Brief prompt | 1,869 | 88.44% | 79.72% | 0.8111 |
| MiniCPM-V | Default prompt | 1,710 | 89.94% | 82.81% | 0.8292 |
| XComposer | Brief prompt | 2,055 | 85.55% | 77.32% | 0.7833 |
| XComposer | Default prompt | 1,955 | 89.41% | 81.07% | 0.8166 |

As shown in Table 5, the stricter threshold lowered accuracy in all conditions, by between about 2 points (GLaMM Default) and 13 points (human descriptions and CogVLM). The overall pattern remained similar: accuracy stayed close across human and model descriptions, and differences between Brief and Default were small and of inconsistent sign across speakers. The ordering is not strictly preserved, however; for instance, Human Spoken is slightly above Human Written at 0.50 and slightly below at 0.75. Excluding GLaMM, whose few trials make its estimates unstable, mean IoU ranged between 0.78 and 0.85 across speakers and conditions. This suggests that the main pattern is not an artefact of the 0.50 threshold.

### B(c). Distractor stratification analysis

Table 6 reports comprehension accuracy by description condition and number of distractors (0 to 4 co-occurring objects of the same category).

Marginal accuracy decreased with visual clutter, going from about 92% (0 distractors) to about 73% (4 distractors). Within each level, differences between conditions were modest: Brief and Default model descriptions differed by at most about 2.4 points in tiers 0 to 3. The human advantage is supported by the pooled analysis with ≥ 1 distractor, where it is significant against Default but not Brief prompts. Tiers 3 and 4 contain fewer human trials, so their estimates should be interpreted with caution.

**Table 6.** Comprehension accuracy for QWEN2.5-VL-7b stratified by number of same-category distractors and description type. The penultimate block reports pooled metrics for all trials containing at least one competitor (≥ 1 distractor), while the final block reports overall trial-weighted accuracy across the full dataset.

| Distractors | Condition | Accuracy (%) | Marginal mean (%) |
| :--- | :--- | :---: | :---: |
| 0 | Human Spoken / Written / Model Brief / Model Default | 94.12 / 94.10 / 91.13 / 93.09 | 92.41 |
| 1 | Human Spoken / Written / Model Brief / Model Default | 87.22 / 87.41 / 82.91 / 82.77 | 84.29 |
| 2 | Human Spoken / Written / Model Brief / Model Default | 86.77 / 88.62 / 83.06 / 80.63 | 84.00 |
| 3 | Human Spoken / Written / Model Brief / Model Default | 84.72 / 81.38 / 80.72 / 83.07 | 82.45 |
| 4 | Human Spoken / Written / Model Brief / Model Default | 73.86 / 71.08 / 77.50 / 70.19 | 72.96 |
| ≥ 1 | Human Spoken / Written / Model Brief / Model Default | 85.35 / 84.90 / 82.33 / 81.56 | 83.09 |
| All | Human Spoken / Written / Model Brief / Model Default | 88.31 / 88.15 / 88.19 / 88.72 | 88.41 |

### B(d). Item-level alignment between discriminativity and comprehension

To assess whether listener accuracy is coupled with pragmatic discriminativity at the individual trial level, we evaluated a matched subset of instances (N = 8,026) for which both comprehension (IoU) and ground-truth human discriminativity labels were available. Because the probing corpus draws on trials used to judge discriminativity, its distribution differs from the full behavioural dataset reported in Table 1, which also includes conditions with no distractor trials at near-ceiling accuracy; this accounts for the lower average accuracy reported below relative to Table 1.

As reported in Table 7, comprehension accuracy drops when the input is ambiguous (37.58%, mean IoU = 0.361) compared to when it is unambiguous (76.91%, mean IoU = 0.692). Welch's t-test on continuous IoU (t = 40.48, p < 0.001) and a chi-square test on thresholded accuracy (χ² = 1208.6, p < 0.001) confirm this contrast. A trial-level logistic regression with image-clustered standard errors demonstrates that pragmatic discriminativity is a robust predictor of localisation success (β = 1.711, SE = 0.056, p < 0.001, OR = 5.53, 95% CI [4.82, 6.35]). This failure on underdiscriminative cases confirms that the listener relies on the referential content of the expression rather than exploiting visual salience or spatial priors. Furthermore, prompt stratification reveals that brevity is advantageous specifically when expressions are discriminative: brief descriptions achieve 79.90% accuracy compared to 72.37% for default descriptions (+7.53 percentage points), while underdiscriminative inputs remain near floor performance regardless of verbosity (36.05% vs. 39.74%).

**Table 7.** QWEN2.5-VL-7b referring expression comprehension performance stratified by ground-truth pragmatic discriminativity. Accuracy denotes trials meeting IoU ≥ 0.50. Human descriptions were not included in this analysis, as there was not a sufficient number of sentences judged as underdiscriminative.

| Pragmatic status | Prompt style | Trials (N) | Accuracy (%) | Mean IoU (SD) | Median IoU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Underdiscriminative (label 0) | All | 4,721 | 37.58 | 0.361 (0.400) | 0.109 |
| Discriminative (label 1) | All | 3,305 | 76.91 | 0.692 (0.330) | 0.854 |
| Underdiscriminative (label 0) | Brief | 2,771 | 36.05 | – | – |
| Discriminative (label 1) | Brief | 1,995 | 79.90 | – | – |
| Underdiscriminative (label 0) | Default | 1,950 | 39.74 | – | – |
| Discriminative (label 1) | Default | 1,310 | 72.37 | – | – |

## C. Layer-wise probing trajectory

To evaluate whether representational separability is localised or distributed across the model layers, we trained and evaluated linear probes across the network depth (sampled at all even layers, plus the final layer 27) for both the Image + Text and Text-Only conditions, on the unseen test set (Figure 5).

Linear decodability increases sharply within the earliest layers (exceeding 0.82 AUROC by layer 2 in both modalities), consistent with prior findings that surface linguistic features and lexical regularities are encoded early in transformer backbones. From layer 16 onward, multimodal representations settle into a robust, stable plateau (AUROC ≈ 0.89), whereas text-only performance fluctuates more across intermediate and late layers. Overall, the multimodal advantage emerges only from the intermediate layers onwards.

*Figure 5.* Layer-wise linear probing AUROC across network depth in QWEN2.5-VL-7b for the Image + Text and Text-Only conditions, computed on the held-out test set. Probes were trained on the training split, and the reference layer (18, red line) was selected on the validation split.

## D. Multimodal vs. unimodal error breakdown

To evaluate the interaction between visual and linguistic modalities in representational probing beyond individual qualitative cases, we cross-tabulated the predictions of the multimodal (Image + Text) and unimodal (Text-Only) linear probes across the unseen test set (N = 1,915 trials).

**Table 8.** Classification breakdown across modalities at layer 18 on the test set (N = 1,915). When the two probes disagree (N = 265), adding visual features resolves the ambiguity in 69.06% of cases, compared with 30.94% where visual cues degrade performance.

| Modality agreement & outcome | Trials (N) | % of test set |
| :--- | :---: | :---: |
| *Agreement* | | |
| Both models correct | 1,342 | 70.08% |
| Both models incorrect | 308 | 16.08% |
| *Discrepancy* | | |
| Vision rescues text (multimodal correct, text-only error) | 183 | 9.56% |
| Vision interferes (text-only correct, multimodal error) | 82 | 4.28% |
| **Total** | **1,915** | **100.0%** |

As shown in Table 8, the two probes agree on 86.16% of all test instances (1,342 joint successes and 308 joint failures). Among the 265 trials exhibiting cross-modal discrepancy (13.84% of the dataset):

- In 69.06% of discrepant cases (N = 183, or 9.56% of all trials), the inclusion of the visual modality rectifies a misclassification made by the text-only probe.
- In 30.94% of discrepant cases (N = 82, or 4.28% of all trials), the multimodal representation fails despite correct classification by the text-only probe, reflecting instances where the two probes disagree in the opposite direction.

McNemar's test confirms that the multimodal probe corrects significantly more unimodal errors than it introduces (χ² = 38.5, p < 0.001). This asymmetric distribution shows that the multimodal probe corrects more unimodal errors than it introduces, consistent with, but not proof of, a corrective role for visual representations at the representational level.

## E. Control for baseline in REC task

This section investigates whether the high accuracy of QWEN2.5-VL-7b reflects sensitivity to what a description says, or reliance on visual salience/positional cues that could allow the target to be found. We ran the model on the original target–description pairs from the RefOI dataset under four conditions, using the same bounding-box prompt as in the main experiment:

- **(i) Original**: the description produced for the target, as in the previous trials.
- **(ii) Mismatched (same category)**: a description of a different target within the same object label, taken from another image and of the same description type (Human Written/Spoken, Model Default/Brief) as the original.
- **(iii) Mismatched (random)**: as in (ii), but the object label can be any from the dataset.
- **(iv) None**: the description is removed from the prompt, which only asks the model to identify the target.

Success was defined as in the main text (IoU ≥ 0.5). Odds ratios come from a logistic regression on the pairs with at least one distractor, with condition as predictor (reference: Original), the number of distractors as a covariate, and standard errors clustered by image identifier.

With at least one same-category distractor, accuracy was 83.09% with the original description, 33.38% with no description and 26.07% with a mismatched same-category description. The odds of success were about twelve times lower with no description and about seventeen times lower with a mismatched same-category description. With a random mismatched description, accuracy was close to zero (all differences p < .001). When the target was the only object of its category (no distractors), accuracy without a description was high (77.42% vs. 92.41% with the original description), as salience is largely sufficient in these scenes, which often feature a single object.

Overall, in scenes with referential competition the model performs close to chance without a description and even worse with a misleading one, showing that it relies on what the description says rather than on salience or positional priors. The similar accuracy across human and model descriptions in the main analysis therefore cannot be attributed to the listener ignoring the description's content.

**Table 9.** Control conditions for QWEN2.5-VL-7b. Accuracy (IoU ≥ 0.5) with 95% bootstrap CIs over images; missing bounding boxes count as errors. OR: odds ratio of success relative to the original description, from a logistic regression on items with ≥ 1 distractor, adjusted for the number of distractors, with standard errors clustered by image identifier; all p < .001.

| Description given | ≥ 1 distractor: accuracy [95% CI] | ≥ 1 distractor: OR [95% CI] | No distractors: accuracy [95% CI] |
| :--- | :---: | :---: | :---: |
| Original | 83.09% [81.5, 84.65] | ref. | 92.41% [91.62, 93.10] |
| Mismatched (same category) | 26.07% [24.24, 27.81] | 0.059 [0.050, 0.069] | 51.21% [49.05, 53.50] |
| Mismatched (random) | 3.47% [2.95, 4.00] | 0.006 [0.005, 0.007] | 2.86% [2.43, 3.30] |
| None | 33.38% [29.41, 37.48] | 0.084 [0.068, 0.102] | 77.42% [73.40, 81.11] |
