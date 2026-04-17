# TFGpractice

Workspace principal del proyecto de predicción de `path_loss` UAV sobre CKM.

## Documentos canónicos

- Historial y estado de los tries:
  - [`VERSIONS.md`](VERSIONS.md)
- Comandos y flujo de cluster:
  - [`cluster/CLUSTER_COMMANDS.md`](cluster/CLUSTER_COMMANDS.md)
- Export y visualización:
  - [`scripts/README_EXPORT_VISUALS.md`](scripts/README_EXPORT_VISUALS.md)

## Línea activa reciente

- [`TFGSeventyThirdTry73/README.md`](TFGSeventyThirdTry73/README.md)
  - 6 expertos, **sin prior**, predicción directa de `path_loss`
- [`TFGSeventyFourthTry74/README.md`](TFGSeventyFourthTry74/README.md)
  - 3 expertos, **sin prior**, banda estrecha de altura `47.5-52.5 m`, sin modulación de altura
- [`TFGSeventyFifthTry75/README.md`](TFGSeventyFifthTry75/README.md)
  - 3 expertos, **sin prior**, todas las alturas, FiLM de altura reactivado, continuación desde `Try 74`

## Plot agregado de tries recientes

- Script:
  - [`cluster/plot_cluster_outputs_try68plus.py`](cluster/plot_cluster_outputs_try68plus.py)
- Salidas:
  - `cluster_outputs/_plots_try68plus/summary_best_val_rmse.csv`
  - `cluster_outputs/_plots_try68plus/summary_best_val_rmse.png`
