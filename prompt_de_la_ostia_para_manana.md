## Este prompt 

mira, sabes que, creame un try 80, ese try 80 va a ser complicado de cojones pero para alguien que procesa datos como tú (la parte difícil) será fácil, va a ser un solo modelo, enorme, que me prediga tanto path loss en nLoS y LoS, delay spread y angular spread.

De inputs tendrá el prior de try 78 de try 78 para LoS y nLoS, y el de try 79 para angular spread y delay spread.

Del try 78, ya que hay fspl_rmse_los_pw, radial_rmse_los_pw y two_ray_rmse_los_pw. Pero el unico sub 2dB es  two_ray_rmse_los_pw, solo implementa ese e importa esa calibración de ahí.

De arquitectura tendra el histograma based de try 76 y 77 (mlp mas UNet y sinusoidal height film conditioning). Que el modelo tenga muchos channels. Como 96. Pero que el split de train, val y test sean como en el try 76 y 77. Dentro del modelo en sí puede tener expertos internos, como veas, pero con el prior no creo que sea necesario. Es importante que le de importancia al prior... No queremos resultados peores que el prior. Piensa como hacerlo en esta arquitectura. Guiate por los .md en ambos tries (78 y 79) de como mejorar resultados con DL. Ten en cuenta que tendran que adaptarse para un megamodelo no por channels y que predice todo a la vez.

Tambien deja documentacion de TODAS las formulas utilizadas para el prior (con citacion) (que vendran del try 77 y 78 ) y tambien de la estructura del modelo, en algun .md. Tambien de la arquitectura final.

Tambien, como en todos los tries, que solo se predigan los pixeles que tienen topology == 0, es decir, donde no hay edificios
Inputs: topology, nLoS y LoS mask y prior. Outputs -> path loss, delay spread y angular spread (juntas pero en el modelo interno separados por LoS y nLoS probablemente)

Tambien ya que estamos deja un script para guardar en un .hdf5 todos los priors precomputados para entrenar más rápido también. Pero que el modelo pueda entrenar y obtener resultados sin tenerlo, que sea opcional.

También deja un script para plottear las losses y errors (rmse es el importante pero tambien kl si quieres) de cada output tanto LoS y nLoS, y como mejoran respecto al prior. En el Json producido por validacion deberia haber rmse global (por pixel, siempre por pixel) de path loss, angular spread y delay spread, rmse dividido por LoS y nLoS de path loss angular spread y delay spread. Y exactamente lo mismo pero separado por expertos (que creo que mejor tener 3 y 3 tipos de antenna en total 9). Tambien exactamente lo mismo, que ya son 3+6+27=36 metricas, 36 más, de lo mismo pero en el prior. Tambien que salga, pero ya no tan desglosado la kl divergergence y map_nll y el resto de losses. Y lo que ha tardado la epoch en entrenar y validacion.

Al acabar y no antes, crea scripts de upload como en try 76 y 77, 1, 2 y 4 gpus. Que puedan depender del anterior

## Para este chat del Codex

"Compare Try 78 to 47"