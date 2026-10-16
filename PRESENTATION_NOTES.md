# Master's project notes: WildfirePINN v1

Suggested name when the slide is about the frozen checkpoint: a level-set neural baseline developed within the WildfirePINN framework. The repository name can stay WildfirePINN. The selected file has no PDE term in its loss, so calling that file a physics-informed model overstates what was frozen.

## Narrative

**Problem.** A fire perimeter is a moving boundary. The project asks whether a neural signed-distance field, trained on satellite detections, can carry that boundary forward in time, and whether a weather-derived spread equation helps.

**Data.** Kincade, October 2019. Twenty VIIRS overpasses, kept in time order: 12 train, 4 validation, 4 test. Terrain and weather exist on the same 90 m grid. They are used in the physics ablation. They are not inputs to v1.

**Persistent fire representation.** Each detection paints a pixel disk. The target at a pass is the union of every accepted disk up to that pass. The region only grows. Gaps that no pixel covered stay unburned. This is a lower-bound burned area, not a surveyed perimeter.

**Level-set neural network.** A 4×128 tanh network maps normalized (time, x, y) to φ. Negative is burned, zero is the perimeter, positive is outside. v1 is trained to match the persistent signed distance on the training passes. Adam, learning rate 0.001, 4,000 epochs, seed 0. PDE weight 0.

**Physics hypothesis.** The same code can add `φ_t + R_eff|∇φ| = 0`, with `R_eff` from fuel, wind, slope, and moisture. The hypothesis was that this constraint would keep the forecast on the observed fire.

**Diagnostic experiments.** The residual is already large on the training masks: median observed front speed is about 0 on ten of eleven intervals, while weather `R_eff` is about 0.05–0.11 m/s. Replacing R with a speed measured from the training masks drops that residual by about 30× and raises validation IoU from 0.404 to 0.471. Untouched test IoU goes from 0.129 to 0.120, and test area error gets worse. A small classifier cannot tell active from inactive perimeter sectors (ROC-AUC 0.581). Most new area is an attached lobe, but pass 9→10 places about half of its 44 km² at least 1 km from the previous fire.

**Final model.** Observation-only level set. Checkpoint hash `93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996`. Chosen on validation. Test was not used to choose it.

**Results.** Mean IoU 0.686 train, 0.404 validation, 0.129 test. A zero contour exists on every pass. Validation area runs about 67 km² hot. On the test passes the predicted region sits south of the persistent fire and overlap falls to IoU 0.08 by the last overpass.

**Limitations.** The target is a sparse satellite union, not a mapped perimeter. The forecast is one fire, one sensor, one split. v1 has no mechanism that prefers spread in the wind direction. Test overlap is too low for an operational perimeter product.

**Future work.** Denser time sampling, or a target that does not omit the corridor between detections. A propagation term is worth another look only with a rate that can be near zero on a stationary flank and fast on a lobe, and only if that gain shows up on a split that was not used to build the rate.

## 30 seconds

We represent the Kincade fire as a signed distance learned from VIIRS pixel footprints. The network is a small fully connected model of position and time. A weather spread equation was tested and then left out: the satellite perimeter is usually stationary between passes, and the equation did not improve the untouched test. The frozen model follows the training fire, reaches IoU 0.40 on validation, and drifts off the fire on the test days. It is a level-set baseline from the WildfirePINN study, not a physics-informed forecast.

## 2 minutes

The practical object is the burned region after each satellite pass. VIIRS gives hot pixels, not a surveyed outline, so the target keeps every accepted pixel disk from the start of the fire through the current pass. That region is monotonic. Most of its boundary does not move from one pass to the next. When area does jump, it is often one lobe several kilometers deep, and on pass 9→10 a large piece appears more than a kilometer from the previous region because the observations are hours apart.

The model is a 50,177-parameter network. Inputs are normalized time and map position. Output is φ, negative inside the predicted fire. Training matches φ to that persistent distance on the first 12 passes. There is no weather input and no PDE weight in the frozen checkpoint.

We did implement the level-set PDE `φ_t + R|∇φ| = 0` and checked it on an analytic circle. On Kincade the same residual is about the size of R, because the observed front speed is usually zero while the weather rate is not. A rate estimated from the training fronts makes the residual small and improves validation IoU from 0.40 to 0.47. The untouched test does not improve: IoU 0.12 versus 0.13, with a larger area error. Weather and terrain also fail a simple test of which part of the perimeter will move next.

So v1 is the observation-only network. Train IoU is 0.69. Validation IoU is 0.40, with the predicted fire about 67 km² larger than the target. Test IoU is 0.13, and the maps show the prediction sliding south of the real fire. The level-set drawing of a perimeter is still the right picture. The weather propagation constraint is not part of the model we would submit.

## Questions an advisor may ask

**Why call this a PINN if the selected checkpoint has no PDE loss?**

For the frozen checkpoint, that name is misleading. Physics-informed means the PDE residual enters the training objective. In v1 that weight is zero. The honest description is a level-set neural baseline developed within the WildfirePINN framework. The framework includes the PDE, the audits, and the ablation. The submitted weights do not.

**Validation IoU improved with the observed-speed PDE. Why not ship that?**

It was the better validation overlap, and it was also the worse validation area error. On the test passes, which were not used to choose it, overlap did not improve and area error roughly doubled. A gain that appears only on the selection split is not the frozen model.

**Is IoU 0.13 a usable forecast?**

No. The contour still exists, and the area is in the right range on the last test passes, but the location is wrong. The deliverable is the evaluated baseline and the reason the physics term was not kept, not an operational perimeter.

**What does φ mean on the map?**

Negative is the model's burned region, zero is the drawn perimeter, positive is unburned. The training target uses the same sign.

**Why not score the hotspot hulls?**

The hull fills concavities the pixels never covered. The persistent union is the observation we chose to trust. `evaluate_real` in the package still scores hulls; the v1 script does not use that path.

**Is pass 9→10 a spot fire, or a bug?**

Neither conclusion is supported. About 21 km² starts at least 1 km from the previous persistent region. The mask never removes burned cells, the distances are in meters, and the time step is in seconds. The missing piece is the hours between overpasses. A disk union can skip the corridor a fire actually crossed.

**Did the network fail to fit the training fire?**

It fits the training passes reasonably: IoU 0.69, boundary distance about 800 m, area error about 10 km². The break is after the training window.
