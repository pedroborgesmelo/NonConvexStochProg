# author: pedro.borges.melo@gmail.com
# date: April/2019

##########################################################################################
# 1) uses parametric differentiability of interior penalty solutions to build an upper smoothing
# 2) just calls functions of my lib of parametric analysis of convex problems
##########################################################################################

PARAMETRIC_ANALYSIS_ENGINE_SRC_PATH = "../../2018-10-parametricAnalysisEngine/src/"

# libs

include(string(PARAMETRIC_ANALYSIS_ENGINE_SRC_PATH, "parametricAnalysisEngine.jl"))

# own includes

include("stochProg.jl")
include("standardLinearStochProg.jl")

