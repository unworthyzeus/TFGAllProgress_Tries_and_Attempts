# Path Loss Model Training From The Literature

This note summarizes how path-loss and radio-map models are usually trained in
the literature, with emphasis on papers that are directly relevant to:

- map-based path-loss prediction;
- radio-map estimation;
- UAV / A2G path-loss learning;
- supervised residual or correction-style pipelines.

The goal is not to list every paper, but to extract the training habits that
appear repeatedly in stronger references.

## Short answer

The most common training recipe is not adversarial and not fully end-to-end
"magic."

The usual pattern is:

1. formulate the task as supervised regression;
2. train on ray-tracing or simulated maps first;
3. fine-tune or adapt using smaller real-measurement datasets;
4. use MSE/NMSE-style losses as the default;
5. split train/test by geography, street, site, or map, not only random pixels;
6. report RMSE in dB and regime-aware generalization metrics.

For hard settings, the papers often add:

- curriculum or two-stage training;
- transfer learning from simulation to measurements;
- feature engineering or physical priors;
- sparse-measurement correction on top of a coarse first predictor.

## Main papers and what they actually do

## 1. RadioUNet

Source:

- Levie et al., "RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks"
  - [arXiv](https://arxiv.org/abs/1911.09002)
  - [HTML](https://ar5iv.labs.arxiv.org/html/1911.09002)

What matters for training:

- The problem is treated as supervised image-to-image regression from map
  geometry and Tx location to a dense path-loss map.
- The paper explicitly says the loss is MSE between predicted radio maps and
  simulation radio maps.
- Training uses Adam.
- Model selection is validation-based.
- For the two-UNet version, training is staged:
  - first UNet trained first;
  - second UNet trained afterward with the first one frozen.
- For adaptation to refined measurements, they use weighted MSE only at the
  measured sparse points and zero weight elsewhere.

Why this matters for us:

- Two-stage training is normal in this literature.
- Sparse correction is usually trained with a masked or weighted regression
  loss, not a GAN-first strategy.
- MSE is still the standard baseline loss for radio-map prediction papers.

Useful lines:

- supervised regression and dataset design:
  - [turn3view0](https://ar5iv.labs.arxiv.org/html/1911.09002)
- curriculum learning and staged training:
  - [turn5view5](https://ar5iv.labs.arxiv.org/html/1911.09002)

## 2. PMNet

Source:

- Lee et al., "PMNet: Robust Pathloss Map Prediction via Supervised Learning"
  - [arXiv](https://arxiv.org/abs/2211.10527)
  - [HTML](https://ar5iv.labs.arxiv.org/html/2211.10527)

What matters for training:

- PMNet is explicitly positioned as supervised learning from ray tracing or
  channel measurements plus map data.
- The dataset is preprocessed before training:
  - interpolation for missing pixels;
  - normalization;
  - outdoor-only handling with buildings excluded as usable prediction region.
- Evaluation is map-based and dense, using NMSE and normalized depth loss
  (NDL), not just a single scalar MAE.
- The paper compares performance under smaller datasets and emphasizes that the
  architecture should still work when data is limited.
- It also points out that RadioUNet used curriculum learning.

Why this matters for us:

- Dense map prediction papers usually normalize aggressively and mask or
  redefine non-usable areas before learning.
- Good papers often evaluate whether the method still works under limited data,
  not just under the largest available train split.
- PMNet is still fundamentally a supervised regression model, not an
  adversarial one.

Useful lines:

- dataset preprocessing and normalization:
  - [turn4view0](https://ar5iv.labs.arxiv.org/html/2211.10527)
- curriculum note when discussing RadioUNet:
  - [turn4view4](https://ar5iv.labs.arxiv.org/html/2211.10527)

## 3. A2G path loss with ray tracing + measurements

Source:

- Li et al., "Air-to-ground path loss prediction using ray tracing and measurement data jointly driven DNN"
  - [ScienceDirect abstract](https://www.sciencedirect.com/science/article/pii/S0140366422003954)

What matters for training:

- The usual recipe is not "measurements only."
- They first train on massive ray-tracing data.
- Then they optimize with a smaller measurement dataset.
- Inputs are interpretable physical variables such as path delay, carrier
  frequency, and reflection angle.

Why this matters for us:

- For UAV/A2G path-loss learning, pretraining on a physics-rich source and then
  fine-tuning on scarce measurements is very normal.
- This is much closer to "coarse prior + learned correction" than to pure
  black-box learning.

Useful lines:

- two-stage RT-then-measurement training:
  - [turn2view2](https://www.sciencedirect.com/science/article/pii/S0140366422003954)

## 4. Real-world radio-map validation

Source:

- Shrestha et al., "Radio Map Estimation: Empirical Validation and Analysis"
  - [arXiv](https://arxiv.org/abs/2310.11036)

What matters for training:

- The paper stresses that most prior work relied too heavily on synthetic data.
- In real data, pure deep estimators do not always justify their complexity.
- Hybrid or enhanced schemes can outperform pure deep models in practice.

Why this matters for us:

- It is a warning against assuming that more architecture automatically solves
  the problem.
- Real-world generalization and data regime matter a lot.
- If the available measured data is limited, simple or hybrid models can still
  be competitive.

Useful lines:

- practical warning about pure deep models:
  - [turn12view0](https://arxiv.org/abs/2310.11036)

## 5. Street-wise generalization on real measurements

Source:

- Gupta et al., "Machine Learning-based Urban Canyon Path Loss Prediction using 28 GHz Manhattan Measurements"
  - [arXiv](https://arxiv.org/abs/2202.05107)
  - [HTML](https://ar5iv.labs.arxiv.org/html/2202.05107)

What matters for training:

- They explicitly criticize random splits that do not test extrapolation well.
- They use street-by-street train/test separation to measure generalization to
  unseen streets.
- They prefer simpler regressors plus strong physical features when data is
  limited.
- They compress raw 3D environment information instead of learning from huge
  raw tensors directly.

Why this matters for us:

- Generalization should be tested on held-out streets, sites, maps, or
  environments whenever possible.
- When data is scarce, a physically engineered feature set plus a simpler
  learner can outperform a heavier model trained on an easier split.
- A paper can reach around 5 dB RMSE, but usually in a narrower and more
  structured setting than full dense urban LoS/NLoS radio-map estimation.

Useful lines:

- street-by-street split and expert features:
  - [turn10view0](https://ar5iv.labs.arxiv.org/html/2202.05107)

## 6. Reciprocity-aware augmentation

Source:

- Dempsey et al., "Reciprocity-Aware Convolutional Neural Networks for Map-Based Path Loss Prediction"
  - [arXiv](https://arxiv.org/abs/2504.03625)

What matters for training:

- They keep the problem supervised.
- The novel ingredient is data augmentation, not a new loss family.
- They add a small number of synthetic reciprocal samples so a model trained on
  downlink drive tests generalizes to uplink / backhaul better.

Why this matters for us:

- Geometry-aware augmentation is a real lever in path-loss work.
- If a data regime underrepresents some link configurations, synthetic but
  physically consistent augmentation can help.

Useful lines:

- augmentation and >8 dB gain on uplink subset:
  - [turn8view0](https://arxiv.org/abs/2504.03625)

## 7. General methodological survey

Source:

- Zhang et al., "Path Loss Prediction Based on Machine Learning: Principle, Method, and Data Expansion"
  - [MDPI](https://www.mdpi.com/2076-3417/9/9/1908)

What matters for training:

- It frames path-loss prediction as supervised regression.
- It lays out the usual pipeline:
  - data collection;
  - feature extraction;
  - feature selection;
  - normalization/scaling;
  - model selection;
  - training/test split;
  - multi-metric evaluation.
- It also explicitly recommends data transfer / expansion when measurements are
  scarce, including:
  - reusing data from related scenarios;
  - reusing data from nearby frequencies;
  - combining ML with classical models.

Why this matters for us:

- This is the cleanest "how are these usually trained?" reference among the
  sources checked.
- It supports the idea that hybrid physical + ML training is standard, not a
  weird hack.

Useful lines:

- supervised regression framing:
  - [turn7view0](https://www.mdpi.com/2076-3417/9/9/1908)

## 8. Transfer learning is not exotic, it is mainstream

Source:

- Lee and Molisch, "A Scalable and Generalizable Pathloss Map Prediction"
  - [arXiv](https://arxiv.org/abs/2312.03950)
  - [IEEE Xplore](https://ieeexplore.ieee.org/document/10682525)

What matters for training:

- This is essentially the extended PMNet line.
- The paper again frames the task as supervised path-loss map prediction from
  map data plus limited RT or measurement labels.
- The important addition is transfer learning:
  - pretrain in one scenario;
  - adapt to a new scenario faster and with less data.
- The authors explicitly report scenario adaptation gains from warm-starting
  rather than retraining from scratch.

Why this matters for us:

- Warm-starting from a strong prior branch or from a nearby propagation domain
  is fully aligned with the literature.
- "Train once on a large synthetic source, then adapt" is one of the most
  common serious recipes in recent radio-map work.

## 9. Challenge papers show what the field now accepts as a fair benchmark

Sources:

- Yapar et al., "The First Pathloss Radio Map Prediction Challenge"
  - [arXiv](https://arxiv.org/abs/2310.07658)
- Bakirtzis et al., "The First Indoor Pathloss Radio Map Prediction Challenge"
  - [arXiv](https://arxiv.org/abs/2501.13698)

What matters for training:

- Both challenge papers are about standardizing datasets, evaluation, and held
  out generalization.
- The important point is methodological:
  - the field benchmarked supervised path-loss map prediction directly;
  - evaluation is challenge-style and scene-aware;
  - winning entries are compared under common RMSE-based metrics.
- This is where the community's practical default becomes visible.

Why this matters for us:

- If the question is "what do people actually train now?", challenge papers
  are very informative.
- They show that the accepted mainstream is still:
  - paired supervision;
  - careful dataset splits;
  - dense map regression;
  - benchmarked generalization across unseen scenes or tasks.

## 10. Recent challenge winners still use supervised map regression

Sources:

- Feng et al., "IPP-Net: A Generalizable Deep Neural Network Model for Indoor Pathloss Radio Map Prediction"
  - [arXiv](https://arxiv.org/abs/2501.06414)
- Li et al., "TransPathNet: A Novel Two-Stage Framework for Indoor Radio Map Prediction"
  - [arXiv](https://arxiv.org/abs/2501.16023)
- Li et al., "RMTransformer: Accurate Radio Map Construction and Coverage Prediction"
  - [arXiv](https://arxiv.org/abs/2501.05190)
- Ghukasyan et al., "Vision Transformers for Efficient Indoor Pathloss Radio Map Prediction"
  - [arXiv](https://arxiv.org/abs/2412.09507)
  - [MDPI](https://www.mdpi.com/2079-9292/14/10/1905)

What matters for training:

- The backbone family has widened:
  - UNet-like;
  - transformer-convolution hybrids;
  - two-stage transformer decoders.
- But the training recipe is still familiar:
  - supervised paired map regression;
  - large synthetic / ray-tracing style supervision;
  - validation on held-out challenge scenes;
  - RMSE-oriented optimization and reporting.
- IPP-Net is explicit about learning from large-scale simulation plus a
  modified 3GPP indoor model.
- TransPathNet is explicit about being a two-stage framework.

Why this matters for us:

- Newer papers are changing architectures more than they are changing the
  training paradigm.
- The field is not saying "throw away supervision and use GANs"; it is saying
  "keep supervision, but use stronger multiscale backbones."

## 11. High-fidelity synthetic EM supervision is common when measurements are scarce

Source:

- Brennan and McGuinness, "Site-specific Deep Learning Path Loss Models based on the Method of Moments"
  - [arXiv](https://arxiv.org/abs/2302.01052)

What matters for training:

- The paper generates synthetic path-loss labels using a Method of Moments
  solver.
- Those synthetic labels are then used to train CNN-based surrogates.
- The point is not just "deep learning works," but that a sufficiently strong
  numerical solver can be used as the teacher.

Why this matters for us:

- This strongly supports the strategy of using physics or deterministic
  simulation as the high-volume source of supervision.
- It also supports the idea that a learned model often acts as a fast surrogate
  for a slower propagation engine.

## 12. Beyond image-to-image papers, tabular path-loss work is still mostly supervised regression

Sources:

- Jo et al., "Path Loss Prediction Based on Machine Learning Techniques: Principal Component Analysis, Artificial Neural Network, and Gaussian Process"
  - [MDPI](https://www.mdpi.com/1424-8220/20/7/1927)
- Kyösti et al., "A Machine Learning Approach for Path Loss Prediction Using Combination of Regression and Classification Models"
  - [MDPI](https://www.mdpi.com/1424-8220/24/17/5855)

What matters for training:

- These papers are not dense map-to-map CNN papers, but they matter because
  they show how path-loss prediction is often trained in simpler measurement
  settings.
- The recurring pattern is:
  - feature engineering;
  - dimensionality reduction or selection;
  - supervised regression;
  - sometimes explicit LOS/NLOS classification before regression.
- The 2024 combination model is especially relevant because it uses:
  - one regressor for LoS;
  - one regressor for NLoS;
  - and a classifier to decide or blend between them.

Why this matters for us:

- Even outside dense radio maps, the standard recipe is still supervised and
  regime-aware.
- Explicit LoS/NLoS branching is not weird; it is a common modeling move.
- This line of literature makes the case that regime decomposition is often
  more natural than forcing one single predictor to handle every condition.

## 13. Graph and physics-informed methods are growing, but they are not yet the baseline default

Sources:

- Bufort et al., "Data-Driven Radio Propagation Modeling using Graph Neural Networks"
  - [arXiv](https://arxiv.org/abs/2501.06236)
- Shahid et al., "ReVeal: A Physics-Informed Neural Network for High-Fidelity Radio Environment Mapping"
  - [arXiv](https://arxiv.org/abs/2502.19646)
- Jia et al., "Physics-Informed Representation Alignment for Sparse Radio-Map Reconstruction"
  - [arXiv](https://arxiv.org/abs/2501.19160)

What matters for training:

- These papers are important because they show where the field is pushing:
  - graph-structured propagation models;
  - PDE-informed or PINN losses;
  - sparse radio-map reconstruction under physical constraints.
- But they are usually framed as:
  - sparse reconstruction;
  - uncertainty-aware reconstruction;
  - or physics-constrained recovery,
  rather than the everyday baseline for paired dense path-loss prediction.

Why this matters for us:

- They are worth reading because they are closer to "what comes next."
- But if the question is "how are these models habitually trained today?",
  they look more like emerging alternatives than the default recipe.

## 14. Adversarial and unpaired training exist, but they are the exception

Sources:

- Qi et al., "ACT-GAN: Radio map construction based on generative adversarial networks with ACT blocks"
  - [arXiv](https://arxiv.org/abs/2401.08976)
- Ma et al., "Radio map estimation using a CycleGAN-based learning framework for 6G wireless communication"
  - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352864825001294)

What matters for training:

- These papers show that GAN-style and unpaired training are genuinely being
  explored.
- The CycleGAN paper is explicit that it addresses the lack of strictly paired
  datasets by using an unpaired setup and a two-stage correction process.
- ACT-GAN emphasizes local texture and radio-map detail.

Why this matters for us:

- This is useful evidence that GANs are not "forbidden" in the field.
- But it is also evidence that they are not the dominant default recipe.
- When adversarial training appears, it is usually presented as a special
  strategy for:
  - unpaired data;
  - texture enhancement;
  - or sparse-detail correction,
  not as the universal baseline.

## What is actually common across the papers

Across these sources, the strongest common habits are:

- Supervised regression is the default.
- MSE / NMSE is still the most common training loss.
- Realistic train/test splitting matters a lot.
- Simulation pretraining plus measurement adaptation is common.
- Transfer learning across scenarios is common.
- Physical or engineered features are often used, not rejected.
- LoS/NLoS decomposition is common in both simple and complex pipelines.
- Sparse correction is usually trained with masked or weighted regression.
- Data augmentation is used when it preserves the geometry or reciprocity of
  the problem.

Less common as the main default:

- GAN-first training.
- Unpaired training.
- Purely physics-informed or GNN-first pipelines as the baseline default.
- Optimizing only on random splits with no geographic holdout.
- Expecting a single monolithic model to solve all LoS and NLoS behavior
  equally well without regime structure.

## Practical implications for us

For our line of work, the literature suggests:

1. Keep the strong LoS prior and dense supervised regression base.
2. Treat NLoS as the main bottleneck, not as a small detail.
3. Prefer residual or correction-style training over full replacement.
4. Evaluate on regime splits and geography-aware splits, not only overall
   random validation.
5. Use physics-guided pretraining or priors whenever measurement data is
   limited.
6. Consider scenario transfer and warm-starting as first-class tools, because
   recent path-loss papers use them explicitly.
7. If stage2 is used, train it as a masked/weighted residual corrector, which
   is closer to the literature than an adversarial path.
8. If the dataset clearly mixes regimes, a regime-aware branch or expert model
   is much easier to justify from the literature than a single monolithic model.

## Bottom line

The literature does not suggest that path-loss models are usually trained with
strong adversarial objectives or purely end-to-end black-box pipelines.

The most common serious recipe is:

- supervised regression;
- careful preprocessing and masking;
- simulation or ray-tracing data as the large source of supervision;
- smaller real-measurement fine-tuning or correction;
- geography-aware validation;
- and physically meaningful inputs or residual structure.

## Source links

- RadioUNet:
  - [https://arxiv.org/abs/1911.09002](https://arxiv.org/abs/1911.09002)
- PMNet:
  - [https://arxiv.org/abs/2211.10527](https://arxiv.org/abs/2211.10527)
- A2G RT + measurement DNN:
  - [https://www.sciencedirect.com/science/article/pii/S0140366422003954](https://www.sciencedirect.com/science/article/pii/S0140366422003954)
- Real-world radio-map validation:
  - [https://arxiv.org/abs/2310.11036](https://arxiv.org/abs/2310.11036)
- Street-wise measurement generalization:
  - [https://arxiv.org/abs/2202.05107](https://arxiv.org/abs/2202.05107)
- Reciprocity-aware augmentation:
  - [https://arxiv.org/abs/2504.03625](https://arxiv.org/abs/2504.03625)
- Survey on ML path-loss training:
  - [https://www.mdpi.com/2076-3417/9/9/1908](https://www.mdpi.com/2076-3417/9/9/1908)
- Transfer-learning PMNet extension:
  - [https://arxiv.org/abs/2312.03950](https://arxiv.org/abs/2312.03950)
- First outdoor challenge:
  - [https://arxiv.org/abs/2310.07658](https://arxiv.org/abs/2310.07658)
- First indoor challenge:
  - [https://arxiv.org/abs/2501.13698](https://arxiv.org/abs/2501.13698)
- IPP-Net:
  - [https://arxiv.org/abs/2501.06414](https://arxiv.org/abs/2501.06414)
- TransPathNet:
  - [https://arxiv.org/abs/2501.16023](https://arxiv.org/abs/2501.16023)
- RMTransformer:
  - [https://arxiv.org/abs/2501.05190](https://arxiv.org/abs/2501.05190)
- Vision transformers for indoor pathloss:
  - [https://arxiv.org/abs/2412.09507](https://arxiv.org/abs/2412.09507)
- Site-specific MoM surrogate:
  - [https://arxiv.org/abs/2302.01052](https://arxiv.org/abs/2302.01052)
- PCA + ANN + GP tabular path-loss modeling:
  - [https://www.mdpi.com/1424-8220/20/7/1927](https://www.mdpi.com/1424-8220/20/7/1927)
- Classification + regression path-loss modeling:
  - [https://www.mdpi.com/1424-8220/24/17/5855](https://www.mdpi.com/1424-8220/24/17/5855)
- GNN radio propagation:
  - [https://arxiv.org/abs/2501.06236](https://arxiv.org/abs/2501.06236)
- ReVeal PINN:
  - [https://arxiv.org/abs/2502.19646](https://arxiv.org/abs/2502.19646)
- Physics-informed sparse radio-map reconstruction:
  - [https://arxiv.org/abs/2501.19160](https://arxiv.org/abs/2501.19160)
- ACT-GAN:
  - [https://arxiv.org/abs/2401.08976](https://arxiv.org/abs/2401.08976)
- CycleGAN radio-map estimation:
  - [https://www.sciencedirect.com/science/article/pii/S2352864825001294](https://www.sciencedirect.com/science/article/pii/S2352864825001294)
