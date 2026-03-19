# TFGFourthTry4 Implementation

`TFGFourthTry4` is a copy of `TFGThirdTry3` for a new experiment branch.

## Origin

- **Base:** `TFGThirdTry3` (full copy)
- **Purpose:** Implement and test **optimizing the fused path-loss map in dB** directly during training

## Planned Change (vs ThirdTry3)

In ThirdTry3, the generator optimizes:
- `recon_loss`: MSE on raw normalized outputs
- `confidence_loss`: auxiliary loss for the confidence head
- `gan_loss`: adversarial term

The validation metric `path_loss_rmse_physical` is computed on the **fused** map (blend of pred + confidence fallback) in dB. These are different targets, so gen_loss can decrease while path_loss_rmse_physical increases.

**In FourthTry4:** Replace the path-loss contribution to the training loss with MSE on the **fused map in dB**. The generator will optimize exactly what we measure.

## Relation to Other Tries

- `TFG_FirstTry1`: original base
- `TFGSecondTry2`: path-loss features
- `TFGThirdTry3`: hybrid confidence + blend fallback
- `TFGFourthTry4`: ThirdTry3 + fused-map-in-dB loss

## Configs and Outputs

Configs are identical to ThirdTry3. Use a distinct `output_dir` when running FourthTry4 experiments (e.g. `outputs/..._fourthtry4/`) to avoid overwriting ThirdTry3 results.
