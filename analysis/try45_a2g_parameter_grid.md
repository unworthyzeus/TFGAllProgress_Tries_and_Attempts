# Try 45 A2G parameter grid search

This search keeps the paper-style A2G functional form and varies only a small number of global coefficients.

Selection rule:

- minimize `0.85 * train_NLoS_RMSE + 0.15 * train_LoS_RMSE`
- train split for search
- validation split only for later comparison

## Best candidate

- params: `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 8.0, "nlos_tau": 4.0}`
- train overall: `35.9119 dB`
- train LoS: `3.7477 dB`
- train NLoS: `59.4783 dB`
- val overall: `34.2503 dB`
- val LoS: `3.8119 dB`
- val NLoS: `59.9437 dB`

## Top 10

1. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 8.0, "nlos_tau": 4.0}` | train NLoS `59.4783` | val NLoS `59.9437` | val overall `34.2503`
2. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 12.0436, "nlos_tau": 4.0}` | train NLoS `59.4797` | val NLoS `59.9468` | val overall `34.2520`
3. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 16.0, "nlos_tau": 4.0}` | train NLoS `59.4811` | val NLoS `59.9498` | val overall `34.2537`
4. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 20.0, "nlos_tau": 4.0}` | train NLoS `59.4825` | val NLoS `59.9528` | val overall `34.2555`
5. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 24.0, "nlos_tau": 4.0}` | train NLoS `59.4839` | val NLoS `59.9559` | val overall `34.2572`
6. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 8.0, "nlos_tau": 6.0}` | train NLoS `59.4844` | val NLoS `59.9566` | val overall `34.2576`
7. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 12.0436, "nlos_tau": 6.0}` | train NLoS `59.4889` | val NLoS `59.9663` | val overall `34.2631`
8. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 8.0, "nlos_tau": 7.52}` | train NLoS `59.4929` | val NLoS `59.9730` | val overall `34.2669`
9. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 16.0, "nlos_tau": 6.0}` | train NLoS `59.4933` | val NLoS `59.9758` | val overall `34.2685`
10. `{"los_log_coeff": -20.0, "los_bias": 0.0, "nlos_bias": -24.0, "nlos_amp": 20.0, "nlos_tau": 6.0}` | train NLoS `59.4978` | val NLoS `59.9855` | val overall `34.2740`
