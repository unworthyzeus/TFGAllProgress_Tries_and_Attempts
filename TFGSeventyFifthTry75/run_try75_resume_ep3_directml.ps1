Set-Location 'C:\TFG\TFGpractice\TFGSeventyFifthTry75'

& 'C:\TFG\.venv\Scripts\python.exe' `
  'train_partitioned_pathloss_expert.py' `
  '--config' `
  'experiments/seventyfifth_try75_experts/try75_expert_allcity_los_local_directml_resume_ep3.yaml'
