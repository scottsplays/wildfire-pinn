# Experiment 2A cause isolation

The 2A checkpoint was scored in place. These runs use the same 64×4 hard-IC network, Adam at 10⁻³, seed 0, and 1,500 epochs. The numerical target is the spatial field R = 0.05 [1 + 0.5 sin(2πx/1500) cos(2πy/1500)]. A midpoint of the saved solver trajectory matches an independent step to 10⁻⁴ m, so the supervised target is the solver field.

## Sampling

On 8,192 uniform collocation points from the trained 2A model:

| region | fraction of samples | share of squared residual | mean φ_t (m/s) | mean \|∇φ\| | corr(residual, R) |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole batch | 1.00 | 1.00 | −0.0511 | 0.997 | 0.979 |
| within 100 m of the numerical front | 0.035 | 0.031 | −0.0505 | 0.999 | 0.968 |
| outside that band | 0.965 | 0.969 | −0.0512 | 0.997 | 0.980 |
| R ≥ 0.05 | 0.504 | 0.427 | −0.0512 | 0.997 | 0.944 |
| R < 0.05 | 0.496 | 0.573 | −0.0511 | 0.997 | 0.943 |
| \|∇R\| above half its sample maximum | 0.803 | 0.674 | −0.0511 | 0.997 | 0.975 |

Mean R is 0.0501 m/s and its standard deviation is 0.0125 m/s. Every uniform point has \|∇φ\| > 0.5. The standard deviation of φ_t is 0.0025 m/s, and its correlation with R is −0.030. The mean squared residual is 1.61×10⁻⁴. The loss of a single speed φ_t = −mean(R) with \|∇φ\| = 1 is mean((R − mean R)²) = 1.55×10⁻⁴.

## Representation

Matching the network output to a target, with the PDE residual removed:

| target | final normalized MSE | final IoU | symmetric distance | field RMSE | growth where R ≥ R0 | growth where R < R0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numerical spatial-R field | 4.85×10⁻⁵ | 0.898 | 32.1 m | 47.7 m | 190.3 m | 189.5 m |
| circle at R = 0.05 m/s | 8.39×10⁻⁷ | 0.982 | 4.54 m | 9.2 m | 204.9 m | 204.4 m |
| 2A PDE run, for scale | — | 0.897 | 32.4 m | 47.9 m | 200.3 m | 201.0 m |

The spatial supervised loss is 5.6×10⁻⁵ at epoch 250 and 4.9×10⁻⁵ at epoch 1,500. On the final numerical front, φ_t for that fit has standard deviation 0.0071 m/s and correlation −0.16 with R. The constant-circle fit has standard deviation 0.0078 m/s and correlation −0.13. The 2A front has standard deviation 0.0009 m/s and correlation −0.16, around φ_t = −0.055 m/s.

Continuing from the spatial supervised weights with the uniform PDE loss for another 1,500 epochs ends at IoU 0.875, symmetric distance 38.4 m, and growth 169.3 m on the fast side and 169.4 m on the slow side.

## Optimization

Parameter gradients are means over four batches of 4,096 points. The speed gap is mean(φ_t | R ≥ R0) − mean(φ_t | R < R0). The two rates in the uniform batch differ by 0.020 m/s, so a resolved front would have a gap near −0.020 m/s.

| state | sample | loss | gradient norm | cosine of high-R and low-R gradients | cosine of loss gradient with the speed gap | gap change from one SGD step (m/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| initialization | uniform | 0.346 | 10.78 | 0.9999 | −0.11 ± 0.42 | 1×10⁻⁴ ± 9×10⁻⁴ |
| trained 2A | uniform | 1.59×10⁻⁴ | 0.0197 | −0.9994 | −0.001 ± 0.37 | −3×10⁻⁷ ± 1×10⁻⁶ |
| trained 2A | front band | 1.37×10⁻⁴ | 0.063 | −0.953 | 0.59 ± 0.03 | −3.4×10⁻⁵ ± 5×10⁻⁶ |

At the trained weights the high-R gradient norm is 0.127 and the low-R norm is 0.166. Their sum has norm 0.039 and their difference has norm 0.293.

One Adam update from the saved 2A optimizer state has step norm 9.3×10⁻⁴ against a weight norm of 23.4. The uniform speed gap changes by −5×10⁻⁶ m/s. The front-band gap changes by −2.0×10⁻⁴ m/s.

## Reading

The front band is 3.5% of a uniform batch, and it contributes 3.1% of the squared residual. High R and low R are each half the batch. The residual–R correlation is 0.98 outside the front as well, because \|∇φ\| stays near 1 and φ_t does not vary. The spatial error is already the volume loss: the trained residual matches the loss of one speed set to the mean of R.

The same optimizer and budget fits the constant-rate circle to 4.5 m. Asked to match the numerical spatial field, it stays 32 m away and the two sides grow 190 m and 189 m. Extra variation in φ_t at the final front is shared with the constant-circle fit and is weakly correlated with R. Further PDE training from that fit keeps one speed.

At initialization the high-R and low-R gradients point the same way, toward a single slower expansion. After training they point in opposite directions and cancel. The uniform loss gradient that remains is a few hundred times smaller, and its alignment with opening a speed gap is indistinguishable from zero across batches. One saved Adam step moves that gap by about 5×10⁻⁶ m/s.
