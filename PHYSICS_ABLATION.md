# Physics ablation

The weather-derived level-set PDE was a research comparison, not a broken training run. WildfirePINN v1 keeps the level-set representation and leaves that propagation constraint out. The numbers below are copied from the saved experiment files.

## Sequence

### 1. Weather-rate level-set PDE

The implemented residual is the outward-growth form

```
φ_t + R_eff |∇φ| = 0
```

with φ negative inside the fire. `R_eff` is the weather spread rate R multiplied by a downwind factor. R depends on fuel, wind speed, slope, and dead-fuel moisture. Wind direction enters only through that factor. An analytic expanding circle at the same speed as R produces a residual of about zero, so the sign, the units, and the chain-rule scaling of the code match the equation. Source: `outputs/kincade_pde_compatibility/summary.json`.

### 2. Persistent burned-region targets

The supervision target is the cumulative union of accepted VIIRS pixel disks, not the concave hull of each overpass. The region only grows. Gaps between pixels stay unburned. Training uses passes 0–11.

### 3. PDE compatibility audit

On the training intervals, inside the 500 m band around the observed front, the median observed normal speed is 0.029 m/s on pass 0→1 and 0 m/s on the other ten intervals. The median weather `R_eff` on those same intervals stays between 0.045 and 0.113 m/s. Mean absolute residual `φ_t + R_eff|∇φ|` is between 0.080 and 0.162 m/s. The observations already disagree with the weather PDE before any network is trained. Source: `outputs/kincade_pde_compatibility/summary.json`, 500 m band.

### 4. Observed-speed experiment

A spread rate estimated only from consecutive training masks, with validation and test fronts excluded, was substituted for `R_eff`. On the near-front training cells the mean absolute residual fell from 0.112 m/s with the weather rate to 0.0032 m/s with the observed speed, a ratio of 0.029. The observed-speed field itself has median 0 and 95th percentile 0.024 m/s. Source: `outputs/kincade_observed_speed_experiment/speed_verification.json`.

Both models keep a zero contour on every validation and test pass. Validation, which was allowed to choose:

| Model | IoU | Dice | Boundary distance | Area MAE |
|---|---|---|---|---|
| Observation only, PDE weight 0 | 0.404 | 0.564 | 1,962 m | 67.0 km² |
| Observed-speed PDE, weight 0.001 | 0.471 | 0.635 | 1,814 m | 83.9 km² |

Untouched test, scored after that choice:

| Model | IoU | Dice | Boundary distance | Area MAE |
|---|---|---|---|---|
| Observation only | 0.129 | 0.225 | 3,919 m | 38.0 km² |
| Observed-speed PDE, weight 0.001 | 0.120 | 0.204 | 4,109 m | 78.6 km² |

The observed-speed term improved validation overlap and made validation area error worse. It did not improve the untouched test. Source: `outputs/kincade_observed_speed_experiment/comparison.json`.

Earlier, among weather-rate weights, validation overlap also preferred observation-only (IoU 0.404) to a weak PDE at weight 0.001 (IoU 0.156) and disqualified weight 0.01 because it kept a zero contour on only half of the validation passes. Source: `outputs/kincade_final_candidates/audit.json`.

### 5. Active-front predictability

Along the old training perimeter, 1,631 of 18,209 samples (9.0%) are locally active: a newly burned cell appears within one diagonal cell. A logistic regression on wind, weather R, slope, moisture, humidity, temperature, and elevation, scored leave-one-interval-out, has ROC-AUC 0.581 and balanced accuracy 0.530. Predicting every point inactive has balanced accuracy 0.500. The available fields do not reliably identify the sectors that move. Source: `outputs/kincade_active_front_diagnostic/summary.json`.

### 6. Front-propagation geometry

For every newly burned training cell, `v_min` is the straight-line distance to the previous persistent polygon divided by the elapsed seconds. Of 192.5 km² of new training area, 83.0% belongs to a component that touches the old region, 4.7% starts between one cell and 1 km away, and 12.4% starts at least 1 km away. Reachable fractions of that new area are 44.4% at 0.05 m/s, 70.4% at 0.10 m/s, and 97.1% at 0.25 m/s.

Pass 9→10 adds 44.1 km² in 11.37 h. The median new cell is 3,679 m from the old region. Only 31.8% of that new area touches the previous fire, and 47.7% is in components whose closest cell is at least 1 km away. A local level-set step would have to cross the gap. The satellite record can omit that corridor, so the gap is sparse observation rather than proof of a separate ignition, and the audit found no mask, distance, or timing bug. Source: `outputs/kincade_front_propagation_audit/summary.json`.

### 7. Decision

v1 is the observation-only checkpoint. The level-set representation remains useful for representing wildfire extent and perimeter evolution, but the tested weather-derived propagation constraint was not retained in WildfirePINN v1 because it was inconsistent with the temporally sparse persistent VIIRS targets and did not improve held-out test performance.
