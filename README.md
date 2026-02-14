# 45-Day Math Course for Deep Learning & Computer Vision

A 45-day sprint that links geometric intuition with practical deep learning and computer vision demos. Each day lives in its own folder under `days/dayXX/` and is anchored by a Jupyter notebook mixing discussion, formulas, and runnable Python. Supporting scripts and generated media sit alongside the notebook so you can reproduce figures or run them headless.

## Daily Workflow

1. Create `days/dayXX/dayXX_topic.ipynb` with all narrative + math tables.
2. Keep reusable helpers in `days/dayXX/code/` (import them inside the notebook or run from CLI).
3. Save derived assets (GIFs, plots, datasets) in `days/dayXX/outputs/`—track only the ones you want on GitHub.
4. Document optional exercises or TODOs at the bottom of the notebook so you can iterate later.

## Repository Layout

```
days/
  day01/
    day01_arrows_and_machines.ipynb
    code/
      arrows_and_machines.py
      visualizations.py
    outputs/
      *.gif
  day02/
    day02_norms_angles.ipynb
    code/
      norms_and_angles.py
      visualizations.py
    outputs/
      *.gif / *.png
  day03/
    day03_gradient_descent.ipynb
    code/
      gradient_descent.py
      visualizations.py
    outputs/
      *.gif / *.png
  day04/
    day04_momentum.ipynb
    code/
      momentum_methods.py
      visualizations.py
    outputs/
      *.gif / *.png
  day05/
    day05_backprop.ipynb
    code/
      backprop_demo.py
      visualizations.py
    outputs/
      *.gif / *.png
  day06/
    day06_landscapes.ipynb
    code/
      landscapes.py
      visualizations.py
    outputs/
      *.gif / *.png
  day07/
    day07_jacobians.ipynb
    code/
      jacobian_demo.py
      visualizations.py
    outputs/
      *.gif / *.png
  day08/
    day08_hessian.ipynb
    code/
      hessian_demo.py
      visualizations.py
    outputs/
      *.gif / *.png
  day09/
    day09_gradients.ipynb
    code/
      gradient_pathologies.py
      visualizations.py
    outputs/
      *.gif / *.png
  day10/
    day10_activations.ipynb
    code/
      activations.py
      visualizations.py
    outputs/
      *.gif / *.png
  day11/
    day11_normalization.ipynb
    code/
      normalization.py
      visualizations.py
    outputs/
      *.gif / *.png
  day12/
    day12_initialization.ipynb
    code/
      initialization.py
      visualizations.py
    outputs/
      *.gif / *.png
  day13/
    day13_pooling.ipynb
    code/
      pooling.py
      visualizations.py
    outputs/
      *.gif / *.png
  day14/
    day14_receptive_fields.ipynb
    code/
      receptive_field.py
      visualizations.py
    outputs/
      *.gif / *.png
  day15/
    day15_padding_stride.ipynb
    code/
      padding_stride.py
      visualizations.py
    outputs/
      *.gif / *.png
  day16/
    day16_dilated_conv.ipynb
    code/
      dilated_conv.py
      visualizations.py
    outputs/
      *.gif / *.png
  day17/
    day17_conv_backprop.ipynb
    code/
      conv_backprop.py
      visualizations.py
    outputs/
      *.gif / *.png
  day18/
    day18_fc_vs_conv.ipynb
    code/
      fc_vs_conv.py
      visualizations.py
    outputs/
      *.gif / *.png
  day19/
    day19_feature_visualization.ipynb
    code/
      feature_visualization.py
      visualizations.py
    outputs/
      *.gif / *.png
  day20/
    day20_modern_cnns.ipynb
    code/
      architecture_summary.py
      visualizations.py
    outputs/
      *.png
  day21/
    day21_encoder_decoder.ipynb
    code/
      encoder_decoder.py
      visualizations.py
    outputs/
      *.png
  day22/
    day22_loss_functions.ipynb
    code/
      losses.py
      visualizations.py
    outputs/
      *.png
  day23/
    day23_segmentation_metrics.ipynb
    code/
      metrics.py
      visualizations.py
    outputs/
      *.png
  day24/
    day24_training_strategies.ipynb
    code/
      training_strategies.py
      visualizations.py
    outputs/
      *.png
  day25/
    day25_postprocessing.ipynb
    code/
      postprocessing.py
      visualizations.py
    outputs/
      *.png
  day26/
    day26_change_detection.ipynb
    code/
      change_detection.py
      visualizations.py
    outputs/
      *.png
  day27/
    day27_multimodal_fusion.ipynb
    code/
      fusion_strategies.py
      visualizations.py
    outputs/
      *.png
  day28/
    day28_eo_foundations.ipynb
    code/
      embeddings_demo.py
      visualizations.py
    outputs/
      *.png
  day29/
    day29_gnn_eo.ipynb
    code/
      gnn_demo.py
      visualizations.py
    outputs/
      *.png
  day30/
    day30_uncertainty_calibration.ipynb
    code/
      uncertainty_calibration.py
      visualizations.py
    outputs/
      *.png
  day31/
    day31_svd.ipynb
    code/
      svd_demo.py
      visualizations.py
    outputs/
      *.png
  day32/
    day32_pca.ipynb
    code/
      pca_demo.py
      visualizations.py
    outputs/
      *.png
  day33/
    day33_rank_nullspace.ipynb
    code/
      rank_nullspace.py
      visualizations.py
    outputs/
      *.png
  day34/
    day34_condition_number.ipynb
    code/
      condition_number.py
      visualizations.py
    outputs/
      *.png
  day35/
    day35_embedding_geometry.ipynb
    code/
      embedding_geometry.py
      visualizations.py
    outputs/
      *.png
  day36/
    day36_limits_continuity.ipynb
    code/
      limits_continuity.py
      visualizations.py
    outputs/
      *.png
  day37/
    day37_partial_derivatives.ipynb
    code/
      partial_derivatives.py
      visualizations.py
    outputs/
      *.png
  day38/
    day38_gradient_vector.ipynb
    code/
      gradient_vector.py
      visualizations.py
    outputs/
      *.png
  day39/
    day39_jacobian.ipynb
    code/
      jacobian_demo.py
      visualizations.py
    outputs/
      *.png
  day40/
    day40_chain_rule.ipynb
    code/
      chain_rule.py
      visualizations.py
    outputs/
      *.png
  day41/
    day41_hessian.ipynb
    code/
      hessian_demo.py
      visualizations.py
    outputs/
      *.png
  day42/
    day42_taylor_expansion.ipynb
    code/
      taylor_demo.py
      visualizations.py
    outputs/
      *.png
  day43/
    day43_critical_points.ipynb
    code/
      critical_points.py
      visualizations.py
    outputs/
      *.png
  day44/
    day44_sgd_saddles.ipynb
    code/
      sgd_saddle.py
      visualizations.py
    outputs/
      *.png
  day45/
    day45_initialization.ipynb
    code/
      initialization.py
      visualizations.py
    outputs/
      *.png
  day46/
    day46_random_variables.ipynb
    code/
      random_variables.py
      visualizations.py
    outputs/
      *.png
  day47/
    day47_expectation_variance.ipynb
    code/
      expectation_variance.py
      visualizations.py
    outputs/
      *.png
  day48/
    day48_mle.ipynb
    code/
      mle_demo.py
      visualizations.py
    outputs/
      *.png
  day49/
    day49_cross_entropy_kl.ipynb
    code/
      info_measures.py
      visualizations.py
    outputs/
      *.png
  day50/
    day50_bias_variance.ipynb
    code/
      bias_variance.py
      visualizations.py
    outputs/
      *.png
  day51/
    day51_regularization_geometry.ipynb
    code/
      regularization_demo.py
      visualizations.py
    outputs/
      *.png
  day52/
    day52_dropout_noise.ipynb
    code/
      dropout_noise.py
      visualizations.py
    outputs/
      *.png
  day53/
    day53_augmentation_invariance.ipynb
    code/
      augmentation_demo.py
      visualizations.py
    outputs/
      *.png
  day54/
    day54_loss_geometry.ipynb
    code/
      loss_geometry.py
      visualizations.py
    outputs/
      *.png
  day55/
    day55_calibration.ipynb
    code/
      calibration_demo.py
      visualizations.py
    outputs/
      *.png
  day56/
    day56_uncertainty_types.ipynb
    code/
      uncertainty_demo.py
      visualizations.py
    outputs/
      *.png
  day57/
    day57_bayesian_thinking.ipynb
    code/
      bayes_demo.py
      visualizations.py
    outputs/
      *.png
  day58/
    day58_information_bottleneck.ipynb
    code/
      information_bottleneck_demo.py
      visualizations.py
    outputs/
      *.png
  day59/
    day59_manifolds_embeddings.ipynb
    code/
      manifold_demo.py
      visualizations.py
    outputs/
      *.png
  day60/
    day60_metric_learning.ipynb
    code/
      metric_learning_demo.py
      visualizations.py
    outputs/
      *.png
  day61/
    day61_self_supervised.ipynb
    code/
      ssl_demo.py
      visualizations.py
    outputs/
      *.png
  day62/
    day62_contrastive_vs_noncontrastive.ipynb
    code/
      ssl_comparison.py
      visualizations.py
    outputs/
      *.png
  day63/
    day63_regularization_revisited.ipynb
    code/
      regularization_revisited.py
      visualizations.py
    outputs/
      *.png
  day64/
    day64_distribution_shift.ipynb
    code/
      distribution_shift_demo.py
      visualizations.py
    outputs/
      *.png
  day65/
    day65_ood_detection.ipynb
    code/
      ood_detection_demo.py
      visualizations.py
    outputs/
      *.png
README.md
```

Future days should mirror this shape, making it easy to navigate the course timeline. Completed notebooks so far:

- `days/day01/day01_arrows_and_machines.ipynb` — geometric storytelling of matrix machines.
- `days/day02/day02_norms_angles.ipynb` — vector norms, angles, cosine similarity, and normalization.
- `days/day03/day03_gradient_descent.ipynb` — gradient descent intuition, learning rate effects, and visualizations.
- `days/day04/day04_momentum.ipynb` — momentum, Nesterov, and inertia-driven optimization stories.
- `days/day05/day05_backprop.ipynb` — chain rule intuition, backprop demo, and gradient-flow visualization.
- `days/day06/day06_landscapes.ipynb` — convex vs non-convex surfaces, curvature, and landscape animations.
- `days/day07/day07_jacobians.ipynb` — Jacobian intuition, local linearization, and sensitivity visualizations.
- `days/day08/day08_hessian.ipynb` — Hessians, curvature intuition, and Newton-vs-GD comparisons.
- `days/day09/day09_gradients.ipynb` — vanishing/exploding gradients, simulations, and mitigation strategies.
- `days/day10/day10_activations.ipynb` — activation geometry, derivatives, and optimization effects.
- `days/day11/day11_normalization.ipynb` — BatchNorm vs LayerNorm, stability intuition, and distribution visualizations.
- `days/day12/day12_initialization.ipynb` — Xavier/He intuition and signal-balance simulations.
- `days/day13/day13_pooling.ipynb` — pooling, downsampling, and hierarchical feature demos.
- `days/day14/day14_receptive_fields.ipynb` — receptive-field growth and multi-scale context visualizations.
- `days/day15/day15_padding_stride.ipynb` — padding/stride geometry demos and stride animations.
- `days/day16/day16_dilated_conv.ipynb` — dilated convolution demos and receptive-field animations.
- `days/day17/day17_conv_backprop.ipynb` — convolution backprop intuition and gradient animations.
- `days/day18/day18_fc_vs_conv.ipynb` — FC vs conv geometry and gradient comparisons.
- `days/day19/day19_feature_visualization.ipynb` — CNN feature maps, filter dreams, and DeepDream-style explorations.
- `days/day20/day20_modern_cnns.ipynb` — modern CNN architecture tour from VGG through ConvNeXt.
- `days/day21/day21_encoder_decoder.ipynb` — UNet/FPN encoder–decoder intuition for dense prediction.
- `days/day22/day22_loss_functions.ipynb` — dense prediction loss functions (CE, Dice, IoU, Focal) and imbalance intuition.
- `days/day23/day23_segmentation_metrics.ipynb` — segmentation metrics (IoU, mIoU, F1, boundary accuracy) and evaluation intuition.
- `days/day24/day24_training_strategies.ipynb` — LR schedules, augmentation, and curriculum strategies for dense prediction.
- `days/day25/day25_postprocessing.ipynb` — post-processing with morphology, connected components, and CRF intuition.
- `days/day26/day26_change_detection.ipynb` — change detection architectures and losses (remote sensing focus).
- `days/day27/day27_multimodal_fusion.ipynb` — optical + SAR + DEM fusion strategies for remote sensing.
- `days/day28/day28_eo_foundations.ipynb` — EO foundation models and embedding-based workflows.
- `days/day29/day29_gnn_eo.ipynb` — graph neural networks for parcels, roads, and spatial relations in EO.
- `days/day30/day30_uncertainty_calibration.ipynb` — uncertainty estimation and calibration for EO decisions.
- `days/day31/day31_svd.ipynb` — SVD geometry and singular value intuition.
- `days/day32/day32_pca.ipynb` — PCA geometry, explained variance, and reconstruction.
- `days/day33/day33_rank_nullspace.ipynb` — rank, null space, and information loss in linear layers.
- `days/day34/day34_condition_number.ipynb` — condition number intuition and numerical stability.
- `days/day35/day35_embedding_geometry.ipynb` — embedding distance, cosine similarity, and collapse detection.
- `days/day36/day36_limits_continuity.ipynb` — limits, continuity, and why gradients exist.
- `days/day37/day37_partial_derivatives.ipynb` — partial derivatives and multivariate slopes.
- `days/day38/day38_gradient_vector.ipynb` — gradient vector geometry and steepest descent.
- `days/day39/day39_jacobian.ipynb` — Jacobian sensitivity and local linearity for vector outputs.
- `days/day40/day40_chain_rule.ipynb` — chain rule and computational graphs for backprop.
- `days/day41/day41_hessian.ipynb` — Hessian curvature, saddles, and second-order intuition.
- `days/day42/day42_taylor_expansion.ipynb` — Taylor expansion and local approximation.
- `days/day43/day43_critical_points.ipynb` — critical points, saddle dominance, and Hessian classification.
- `days/day44/day44_sgd_saddles.ipynb` — saddle dominance and why SGD noise helps.
- `days/day45/day45_initialization.ipynb` — symmetry breaking and variance-stable initialization.
- `days/day46/day46_random_variables.ipynb` — random variables, distributions, and statistical learning intuition.
- `days/day47/day47_expectation_variance.ipynb` — expectation, variance, and concentration intuition for averaging.
- `days/day48/day48_mle.ipynb` — maximum likelihood estimation and why losses are log-likelihoods.
- `days/day49/day49_cross_entropy_kl.ipynb` — cross-entropy, KL divergence, and information mismatch intuition.
- `days/day50/day50_bias_variance.ipynb` — bias–variance tradeoff and generalization intuition.
- `days/day51/day51_regularization_geometry.ipynb` — regularization geometry, L1/L2 constraints, and weight decay.
- `days/day52/day52_dropout_noise.ipynb` — dropout, noise, and implicit regularization for generalization.
- `days/day53/day53_augmentation_invariance.ipynb` — data augmentation and invariance-based regularization.
- `days/day54/day54_loss_geometry.ipynb` — margin losses, robust losses, and geometry-focused loss design.
- `days/day55/day55_calibration.ipynb` — calibration, reliability, and proper scoring rules.
- `days/day56/day56_uncertainty_types.ipynb` — aleatoric vs epistemic uncertainty and decision implications.
- `days/day57/day57_bayesian_thinking.ipynb` — Bayesian priors, posteriors, and MAP intuition.
- `days/day58/day58_information_bottleneck.ipynb` — information bottleneck, compression, and representation learning.
- `days/day59/day59_manifolds_embeddings.ipynb` — manifolds, embeddings, and low-dimensional data structure.
- `days/day60/day60_metric_learning.ipynb` — metric learning, contrastive losses, and similarity geometry.
- `days/day61/day61_self_supervised.ipynb` — self-supervised learning and structure-derived supervision.
- `days/day62/day62_contrastive_vs_noncontrastive.ipynb` — contrastive vs non-contrastive SSL and collapse avoidance geometry.
- `days/day63/day63_regularization_revisited.ipynb` — unified view of regularization via noise, dropout, and augmentation.
- `days/day64/day64_distribution_shift.ipynb` — distribution shift types and domain adaptation intuition.
- `days/day65/day65_ood_detection.ipynb` — OOD detection, confidence failure, and geometry-based signals.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

Launch notebooks with `jupyter lab` or `jupyter notebook` from the repo root so relative imports (e.g., `days.day01.code`) resolve correctly.

## Running Day 1 Code Headless

```bash
# Day 1
python -m days.day01.code.arrows_and_machines
python -m days.day01.code.visualizations   # GIFs → days/day01/outputs/

# Day 2
python -m days.day02.code.norms_and_angles
python -m days.day02.code.visualizations   # GIFs/PNGs → days/day02/outputs/

# Day 3
python -m days.day03.code.gradient_descent
python -m days.day03.code.visualizations   # GIFs/PNGs → days/day03/outputs/

# Day 4
python -m days.day04.code.momentum_methods
python -m days.day04.code.visualizations   # GIFs/PNGs → days/day04/outputs/

# Day 5
python -m days.day05.code.backprop_demo
python -m days.day05.code.visualizations   # GIFs/PNGs → days/day05/outputs/

# Day 6
python -m days.day06.code.landscapes
python -m days.day06.code.visualizations   # GIFs/PNGs → days/day06/outputs/

# Day 7
python -m days.day07.code.jacobian_demo
python -m days.day07.code.visualizations   # GIFs/PNGs → days/day07/outputs/

# Day 8
python -m days.day08.code.hessian_demo
python -m days.day08.code.visualizations   # GIFs/PNGs → days/day08/outputs/

# Day 9
python -m days.day09.code.gradient_pathologies
python -m days.day09.code.visualizations   # GIFs/PNGs → days/day09/outputs/

# Day 10
python -m days.day10.code.activations
python -m days.day10.code.visualizations   # GIFs/PNGs → days/day10/outputs/

# Day 11
python -m days.day11.code.normalization
python -m days.day11.code.visualizations   # GIFs/PNGs → days/day11/outputs/

# Day 12
python -m days.day12.code.initialization
python -m days.day12.code.visualizations   # GIFs/PNGs → days/day12/outputs/

# Day 13
python -m days.day13.code.pooling
python -m days.day13.code.visualizations   # GIFs/PNGs → days/day13/outputs/

# Day 14
python -m days.day14.code.receptive_field
python -m days.day14.code.visualizations   # GIFs/PNGs → days/day14/outputs/

# Day 15
python -m days.day15.code.padding_stride
python -m days.day15.code.visualizations   # GIFs/PNGs → days/day15/outputs/

# Day 16
python -m days.day16.code.dilated_conv
python -m days.day16.code.visualizations   # GIFs/PNGs → days/day16/outputs/

# Day 17
python -m days.day17.code.conv_backprop
python -m days.day17.code.visualizations   # GIFs/PNGs → days/day17/outputs/

# Day 18
python -m days.day18.code.fc_vs_conv
python -m days.day18.code.visualizations   # GIFs/PNGs → days/day18/outputs/

# Day 19
python -m days.day19.code.feature_visualization
python -m days.day19.code.visualizations   # GIFs/PNGs → days/day19/outputs/

# Day 20
python -m days.day20.code.architecture_summary
python -m days.day20.code.visualizations   # PNG plots → days/day20/outputs/

# Day 21
python -m days.day21.code.encoder_decoder
python -m days.day21.code.visualizations   # PNG diagrams → days/day21/outputs/

# Day 22
python -m days.day22.code.losses
python -m days.day22.code.visualizations   # PNG plots → days/day22/outputs/

# Day 23
python -m days.day23.code.metrics
python -m days.day23.code.visualizations   # PNG plots → days/day23/outputs/

# Day 24
python -m days.day24.code.training_strategies
python -m days.day24.code.visualizations   # PNG plots → days/day24/outputs/

# Day 25
python -m days.day25.code.postprocessing
python -m days.day25.code.visualizations   # PNG plots → days/day25/outputs/

# Day 26
python -m days.day26.code.change_detection
python -m days.day26.code.visualizations   # PNG plots → days/day26/outputs/

# Day 27
python -m days.day27.code.fusion_strategies
python -m days.day27.code.visualizations   # PNG plots → days/day27/outputs/

# Day 28
python -m days.day28.code.embeddings_demo
python -m days.day28.code.visualizations   # PNG plots → days/day28/outputs/

# Day 29
python -m days.day29.code.gnn_demo
python -m days.day29.code.visualizations   # PNG plots → days/day29/outputs/

# Day 30
python -m days.day30.code.uncertainty_calibration
python -m days.day30.code.visualizations   # PNG plots → days/day30/outputs/

# Day 31
python -m days.day31.code.svd_demo
python -m days.day31.code.visualizations   # PNG plots → days/day31/outputs/

# Day 32
python -m days.day32.code.pca_demo
python -m days.day32.code.visualizations   # PNG plots → days/day32/outputs/

# Day 33
python -m days.day33.code.rank_nullspace
python -m days.day33.code.visualizations   # PNG plots → days/day33/outputs/

# Day 34
python -m days.day34.code.condition_number
python -m days.day34.code.visualizations   # PNG plots → days/day34/outputs/

# Day 35
python -m days.day35.code.embedding_geometry
python -m days.day35.code.visualizations   # PNG plots → days/day35/outputs/

# Day 36
python -m days.day36.code.limits_continuity
python -m days.day36.code.visualizations   # PNG plots → days/day36/outputs/

# Day 37
python -m days.day37.code.partial_derivatives
python -m days.day37.code.visualizations   # PNG plots → days/day37/outputs/

# Day 38
python -m days.day38.code.gradient_vector
python -m days.day38.code.visualizations   # PNG plots → days/day38/outputs/

# Day 39
python -m days.day39.code.jacobian_demo
python -m days.day39.code.visualizations   # PNG plots → days/day39/outputs/

# Day 40
python -m days.day40.code.chain_rule
python -m days.day40.code.visualizations   # PNG plots → days/day40/outputs/

# Day 41
python -m days.day41.code.hessian_demo
python -m days.day41.code.visualizations   # PNG plots → days/day41/outputs/

# Day 42
python -m days.day42.code.taylor_demo
python -m days.day42.code.visualizations   # PNG plots → days/day42/outputs/

# Day 43
python -m days.day43.code.critical_points
python -m days.day43.code.visualizations   # PNG plots → days/day43/outputs/

# Day 44
python -m days.day44.code.sgd_saddle
python -m days.day44.code.visualizations   # PNG plots → days/day44/outputs/

# Day 45
python -m days.day45.code.initialization
python -m days.day45.code.visualizations   # PNG plots → days/day45/outputs/

# Day 46
python -m days.day46.code.random_variables
python -m days.day46.code.visualizations   # PNG plots → days/day46/outputs/

# Day 47
python -m days.day47.code.expectation_variance
python -m days.day47.code.visualizations   # PNG plots → days/day47/outputs/

# Day 48
python -m days.day48.code.mle_demo
python -m days.day48.code.visualizations   # PNG plots → days/day48/outputs/

# Day 49
python -m days.day49.code.info_measures
python -m days.day49.code.visualizations   # PNG plots → days/day49/outputs/

# Day 50
python -m days.day50.code.bias_variance
python -m days.day50.code.visualizations   # PNG plots → days/day50/outputs/

# Day 51
python -m days.day51.code.regularization_demo
python -m days.day51.code.visualizations   # PNG plots → days/day51/outputs/

# Day 52
python -m days.day52.code.dropout_noise
python -m days.day52.code.visualizations   # PNG plots → days/day52/outputs/

# Day 53
python -m days.day53.code.augmentation_demo
python -m days.day53.code.visualizations   # PNG plots → days/day53/outputs/

# Day 54
python -m days.day54.code.loss_geometry
python -m days.day54.code.visualizations   # PNG plots → days/day54/outputs/

# Day 55
python -m days.day55.code.calibration_demo
python -m days.day55.code.visualizations   # PNG plots → days/day55/outputs/

# Day 56
python -m days.day56.code.uncertainty_demo
python -m days.day56.code.visualizations   # PNG plots → days/day56/outputs/

# Day 57
python -m days.day57.code.bayes_demo
python -m days.day57.code.visualizations   # PNG plots → days/day57/outputs/

# Day 58
python -m days.day58.code.information_bottleneck_demo
python -m days.day58.code.visualizations   # PNG plots → days/day58/outputs/

# Day 59
python -m days.day59.code.manifold_demo
python -m days.day59.code.visualizations   # PNG plots → days/day59/outputs/

# Day 60
python -m days.day60.code.metric_learning_demo
python -m days.day60.code.visualizations   # PNG plots → days/day60/outputs/

# Day 61
python -m days.day61.code.ssl_demo
python -m days.day61.code.visualizations   # PNG plots → days/day61/outputs/

# Day 62
python -m days.day62.code.ssl_comparison
python -m days.day62.code.visualizations   # PNG plots → days/day62/outputs/

# Day 63
python -m days.day63.code.regularization_revisited
python -m days.day63.code.visualizations   # PNG plots → days/day63/outputs/

# Day 64
python -m days.day64.code.distribution_shift_demo
python -m days.day64.code.visualizations   # PNG plots → days/day64/outputs/

# Day 65
python -m days.day65.code.ood_detection_demo
python -m days.day65.code.visualizations   # PNG plots → days/day65/outputs/
```

Each notebook (`days/dayXX/*.ipynb`) walks through the same math with commentary and optional animation toggles.

## Git & GitHub Workflow

1. **Initialize (first time only)**
   ```bash
   git init
   git add .
   git commit -m "Day 1: Arrows & Machines"
   git branch -M main
   git remote add origin git@github.com:<your-user>/math_course_dlcv.git
   ```
2. **Sync changes** after editing notebooks or code
   ```bash
   git add days/day01 day01_arrows_and_machines.ipynb README.md
   git commit -m "Update Day 1 notebook"   # use a descriptive message
   git push -u origin main                  # first push
   # subsequent pushes
   git push
   ```
3. **Pull updates** before starting a new day
   ```bash
   git pull --rebase origin main
   ```
4. **New days**
   ```bash
   cp -R days/day01 days/day02   # or scaffold fresh folders
   git checkout -b day02
   ```

If you prefer HTTPS, replace the SSH remote URL accordingly. Keep large binaries out of Git (consider Git LFS) and avoid committing `.venv/` or `__pycache__/`—they are ignored via `.gitignore`.
