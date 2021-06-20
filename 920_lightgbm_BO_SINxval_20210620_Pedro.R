
#Optimizacion Bayesiana de hiperparametros de LightGBM
#Sin cross validation
#Se testea en 201905 , o sea testeao en un mes del medio !
#Se entrena en TODOS los meses que tienen clase, excepto 201905
#Se utiliza  undersampling de los CONTINUA  al 10%
##Pedro: se le saca optimizacion Bayesiana, se define rango fijo para que haga calculo puntual

#limpio la memoria
rm( list=ls() )
gc()

require("data.table")
require("rlist")
require("yaml")

require("lightgbm")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

#para poder usarlo en la PC y en la nube
switch ( Sys.info()[['sysname']],
         Windows = { directory.root   <-  "M:\\" },
         Darwin  = { directory.root   <-  "~/dm/" },  #Apple MAC
         Linux   = { directory.root   <-  "~/buckets/b1/" } 
       )

#defino la carpeta donde trabajo
setwd( directory.root )

kfinalize <- FALSE
kexperimento  <- 1010

kscript           <- "lightgbm_BO_SINxval"
karch_generacion  <- "./datasets/paquete_premium_exthist_corregido.txt.gz"
kBO_iter    <-  5   #cantidad de iteraciones de la Optimizacion Bayesiana

hs <- makeParamSet( 
         makeNumericParam("learning_rate",    lower= 0.01 , upper=    0.01),
         makeNumericParam("feature_fraction", lower= 0.1  , upper=    0.1),
         makeIntegerParam("min_data_in_leaf", lower= 400    , upper= 400),
         makeIntegerParam("num_leaves",       lower=740L   , upper= 740L),
         makeNumericParam("max_bin",          lower=31,     upper=   31)
        )

#------------------------------------------------------------------------------

get_experimento  <- function()
{
  if( !file.exists( "./maestro.yaml" ) )  cat( file="./maestro.yaml", "experimento: 1000" )
  
  exp  <- read_yaml( "./maestro.yaml" )
  experimento_actual  <- exp$experimento
  
  exp$experimento  <- as.integer(exp$experimento + 1)
  Sys.chmod( "./maestro.yaml", mode = "0644", use_umask = TRUE)
  write_yaml( exp, "./maestro.yaml" )
  Sys.chmod( "./maestro.yaml", mode = "0444", use_umask = TRUE) #dejo el archivo readonly
  
  return( experimento_actual )
}
#------------------------------------------------------------------------------

if( is.na(kexperimento ) )   kexperimento <- get_experimento()  #creo el experimento

#en estos archivos queda el resultado
kbayesiana  <- paste0("./work/E",  kexperimento, "_lightgbm.RDATA" )
klog        <- paste0("./work/E",  kexperimento, "_lightgbm_log.txt" )
kimp        <- paste0("./work/E",  kexperimento, "_lightgbm_importance_" )
kmbo        <- paste0("./work/E",  kexperimento, "_lightgbm_mbo.txt" )
kmejor      <- paste0("./work/E",  kexperimento, "_lightgbm_mejor.yaml" )
kkaggle     <- paste0("./kaggle/E",kexperimento, "_lightgbm_kaggle_prob_" )

#------------------------------------------------------------------------------

loguear  <- function( reg, pscript, parch_generacion, arch)
{
  if( !file.exists(  arch ) )
  {
    linea  <- paste0( "script\tdataset\tfecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )

    cat( linea, file=arch )
  }

  linea  <- paste0( pscript, "\t",
                    parch_generacion, "\t",
                    format(Sys.time(), "%Y%m%d %H%M%S"),
                    "\t",
                    gsub( ", ", "\t", toString( reg ) ),
                    "\n" )

  cat( linea, file=arch, append=TRUE )
}
#------------------------------------------------------------------------------
#esta es la funcion de ganancia, que se busca optimizar
#se usa internamente a LightGBM
#se calcula internamente la mejor ganancia para todos los puntos de corte posibles

fganancia_logistic_lightgbm   <- function(probs, data) 
{
  vlabels  <- getinfo(data, "label")
  vpesos   <- getinfo(data, "weight")

  #solo sumo 29250 si vpesos > 1, hackeo
  tbl  <- as.data.table( list( "prob"=probs, "gan"= ifelse( vlabels==1 & vpesos > 1, 29250, -750 ) ) )

  setorder( tbl, -prob )
  tbl[ , gan_acum :=  cumsum( gan ) ]
  gan  <- max( tbl$gan_acum )

  return(  list( name= "ganancia", value=  gan, higher_better= TRUE ) )
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#funcion que va a optimizar la Bayesian Optimization

EstimarGananciaLightGBM  <- function( x )
{
  set.seed( 999983 )
  modelo  <- lgb.train( data= dtrain,
                        objective= "binary",
                        metric= "custom",
                        valids= list( valid= dvalid ),
                        eval= fganancia_logistic_lightgbm,
                        metric= "custom",  #ATENCION   tremendamente importante
                        first_metric_only= TRUE,
                        boost_from_average= TRUE,
                        feature_pre_filter= FALSE,
                        early_stopping_rounds= as.integer(50 + 5/x$learning_rate),
                        num_iterations= 99999,   #un numero muy grande
                        max_bin= as.integer(x$max_bin),
                        learning_rate= x$learning_rate,
                        feature_fraction= x$feature_fraction,
                        num_leaves=  x$num_leaves,
                        min_data_in_leaf= as.integer( x$min_data_in_leaf ),
                        lambda_l1= 0,
                        lambda_l2= 0,
                        verbosity= -1,
                        verbose= -1,
                        seed= 999983
                      )


  mejor_iter  <- modelo$best_iter
  ganancia    <- unlist(modelo$record_evals$valid$ganancia$eval)[ mejor_iter ] 

  ganancia_normalizada  <-  ganancia
  attr(ganancia_normalizada ,"extras" )  <- list("num_iterations"= modelo$best_iter)  #esta es la forma de devolver un parametro extra

  xx  <- x
  xx$best_iter  <-  modelo$best_iter
  xx$ganancia   <-  ganancia_normalizada
  loguear( xx, kscript, karch_generacion, klog )

  return( ganancia_normalizada )
}
#------------------------------------------------------------------------------

#cargo el dataset
dataset  <- fread(karch_generacion)

campos_lags  <- setdiff(  colnames(dataset) ,  c("clase_ternaria","clase01", "numero_de_cliente","foto_mes") )

#Hago feature Engineering en este mismo script
#agreglo los lags de orden 1
setorderv( dataset, c("numero_de_cliente","foto_mes") )
dataset[,  paste0( campos_lags, "_lag1") :=shift(.SD, 1, NA, "lag"), by=numero_de_cliente, .SDcols= campos_lags]

#agrego los deltas de los lags, con un "for" nada elegante
for( vcol in campos_lags )
{
   dataset[,  paste0(vcol, "_delta1") := get( vcol)  - get(paste0( vcol, "_lag1"))]
}


#creo la clase_binaria especial  1 = { BAJA+2, BAJA+1 }
dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]


campos_buenos  <- setdiff( colnames(dataset) , c("clase_ternaria", "clase01") )

#Entreno en todos los datos que tienen clase 
#valido en 201905, un mes del pasado. Se estaran abriendo las puertas del infierno overfitero?
dataset[  , azar := runif( nrow(dataset)) ]
#Me quedo solo con el 10% de los continua
dataset[  foto_mes<=202003 & foto_mes>=201912 & foto_mes!=201905 & ( clase01==1 | azar < 0.1) ,  generacion:= 1L ]  #donde entreno
dataset[  , azar := NULL ]

dataset[  foto_mes== 201905 ,  validacion:= 1L ]  #donde valido

dataset[  foto_mes== 202005 ,  aplicacion:= 1L ]  #donde aplico el modelo

#genero el formato requerido por LightGBM
dtrain  <- lgb.Dataset( data=  data.matrix(  dataset[ generacion==1, campos_buenos, with=FALSE]),
                        label= dataset[ generacion==1, clase01],
                        weight=  dataset[ generacion==1, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)],
                        free_raw_data= TRUE
                      )

dvalid  <- lgb.Dataset( data=  data.matrix(  dataset[ validacion==1, campos_buenos, with=FALSE]),
                        label= dataset[ validacion==1, clase01],
                        weight=  dataset[ validacion==1, ifelse(clase_ternaria=="BAJA+2", 1.0000001, 1.0)],
                        free_raw_data= TRUE
                      )


#Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar <-  EstimarGananciaLightGBM

configureMlr(show.learner.output = FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
obj.fun  <- makeSingleObjectiveFunction(
              fn= funcion_optimizar,
              minimize= FALSE,   #estoy Maximizando la ganancia
              noisy=    TRUE,
              par.set= hs,   #los hiperparametros que quiero optimizar, definidos al inicio del script
              has.simple.signature= FALSE
             )


ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI())

surr.km  <- makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace = TRUE))


if( kfinalize )
{
  mboFinalize(kbayesiana)
}

#Esta es la corrida en serio
if( !file.exists(kbayesiana) )
{ 
  run  <- mbo(obj.fun, learner= surr.km, control= ctrl) 
} else {
  run  <- mboContinue( kbayesiana )
}

#ordeno las corridas
tbl  <- as.data.table(run$opt.path)
tbl[ , iteracion := .I ]  #le pego el numero de iteracion
setorder( tbl, -y )

#agrego info que me viene bien
tbl[ , script          := kscript ]
tbl[ , arch_generacion := karch_generacion ]

fwrite(  tbl, file=kmbo, sep="\t" )   #grabo TODA la corrida
write_yaml( tbl[1], kmejor )          #grabo el mejor

#------------------------------------------------------------------------------
#Me quedo con el mejor modelo

modelito  <- 1

x  <- tbl[modelito]   #en x quedaron los MEJORES hiperparametros

set.seed( 999983 )
modelo  <- lightgbm( data= dtrain,
                     objective= "binary",
                     metric= "custom",
                     first_metric_only= TRUE,
                     boost_from_average= TRUE,
                     feature_pre_filter= FALSE,
                     nround= x$num_iterations,
                     max_bin= as.integer(x$max_bin),
                     learning_rate= x$learning_rate,
                     feature_fraction= x$feature_fraction,
                     num_leaves=  x$num_leaves,
                     min_data_in_leaf= as.integer( x$min_data_in_leaf ),
                     lambda_l1= 0,
                     lambda_l2= 0,
                     seed= 999983
                   )

#importancia de variables
tb_importancia  <- lgb.importance( model= modelo )
fwrite( tb_importancia, 
        file= paste0( kimp, modelito, ".txt" ), 
        sep="\t" )


#genero el vector con la prediccion, la probabilidad de ser positivo
prediccion  <- predict( modelo, 
                        data.matrix( dataset[ aplicacion==1 , campos_buenos, with=FALSE]))

dataset[ aplicacion==1, prob_pos := prediccion]

#pruebo el intervalo de probabilidades de 0.135 a 0.145 ya que estoy haciendo undersampling
#y ademas los POS={ "BAJA+2", "BAJA+1" }

for( prob in   135:145  )
{
  dataset[ aplicacion==1, estimulo := as.numeric(prob_pos >  prob/1000) ]

  entrega  <- dataset[  aplicacion==1, list( numero_de_cliente, estimulo)  ]

  #genero el archivo para Kaggle
  fwrite( entrega, 
          file= paste0(kkaggle, prob, ".csv"),
          sep=  "," )
}

quit( save="no" )


