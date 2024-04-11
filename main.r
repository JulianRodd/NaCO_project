
# devtools::install_github("richfitz/storr", dependencies=TRUE)
# devtools::install_github("richfitz/remake", dependencies=TRUE)

# install.packages("pandoc")
# install.packages("rmarkdown")

setwd("plant_paper-master")
library("devtools")

install.packages("remotes")

remotes::install_github("richfitz/remake")
library("remake")
remake::install_missing_packages("remake.yml")

install.packages("nleqslv")
install.packages("emutls_w")
install.packages("gfortran")
devtools::install_github("traitecoevo/plant")
devtools::install_github("richfitz/sowsear")

remake::make()
