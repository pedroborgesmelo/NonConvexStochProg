# author: pedro.borges.melo@gmail.com
# date: April/2019

using Base.Test

include("../src/stochProgUpperSmoothing.jl")

###############################################################
# create a stoch prog with only one stage and no scenario problems just to check
# 
# the root problem is: 
# 
#   min -x
#   s.t. x >= 0.0 and x <= 1.0
###############################################################

function test01()
	# build main prob
	name_ = "mainProbFirstStage"
	totFirstStageDecision = 1
	totalParameters_ = 0
	mainProbFirstStage = convexParametricOptProblem(
		name_, totFirstStageDecision, totalParameters_)
	# variables
	x = convert(Array{SymEngine.Basic, 1}, [symbols("x_$i") for i in 1:totFirstStageDecision])
	p = convert(Array{SymEngine.Basic, 1}, [symbols("p_$i") for i in 1:totalParameters_])
	# obj first stage prob: -x_1
	objFuncTmp = -x[1]
	objFuncTmp = paramFuncSymEngine(objFuncTmp, x, p)
	addObjFunctionWithoutBuiltCbk(mainProbFirstStage, objFuncTmp)
	# create constraint: x_1 <= 1.0
	ineqFun = x[1] - 1.0
	ineqFun = paramFuncSymEngine(ineqFun, x, p)
	addInequalityWithoutBuiltCbk(mainProbFirstStage, ineqFun)
	# create stoch prog
	myStochProg = stochProg(mainProbFirstStage)
	# eps	
	epsilonIter = 0.1
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# check obj
	@test norm(optObj + 1) <= 0.0001
	# check sol
	@test norm(finalPrimalSol - [1.0]) <= 0.0001
end

###############################################################
# now we just stack problems independtly in the second stage
# 
# the root problem is: 
# 
#   min -x + 0.5*Q_1(x) + 0.5*Q_2(x)
#   s.t. x >= 0.0 and x <= 1.0
#
# the subproblem is:
# 
#   Q_i(x) = min {-y s.t. y \in [0,1]}
#
# opt val at root: -2.0
# stochastic sol: [1.0]
###############################################################

function test02()
	# build main prob
	name_ = "mainProbFirstStage"
	totFirstStageDecision = 1
	totalParameters_ = 0
	mainProbFirstStage = convexParametricOptProblem(
		name_, totFirstStageDecision, totalParameters_)
	# variables
	x = convert(Array{SymEngine.Basic, 1}, [symbols("x_$i") for i in 1:totFirstStageDecision])
	p = convert(Array{SymEngine.Basic, 1}, [symbols("p_$i") for i in 1:totalParameters_])
	# obj first stage prob: -x_1
	objFuncTmp = -x[1]
	objFuncTmp = paramFuncSymEngine(objFuncTmp, x, p)
	addObjFunctionWithoutBuiltCbk(mainProbFirstStage, objFuncTmp)
	# create constraint: x_1 <= 1.0
	ineqFun = x[1] - 1.0
	ineqFun = paramFuncSymEngine(ineqFun, x, p)
	addInequalityWithoutBuiltCbk(mainProbFirstStage, ineqFun)
	# create stoch prog
	myStochProg = stochProg(mainProbFirstStage)
	# create subproblem
	nameSubProb = "subProb"
	totSecondStageDecisions = 1
	subProb = convexParametricOptProblem(
		nameSubProb, totSecondStageDecisions, totFirstStageDecision)
	# create vars for sub prob
	x2 = [symbols("x_$i") for i in 1:totSecondStageDecisions]
	x1 = [symbols("p_$i") for i in 1:totFirstStageDecision]
	# create objective for subprob
	objFuncTmp = -x2[1]
	objFuncTmp = paramFuncSymEngine(objFuncTmp, x2, x1)
	addObjFunctionWithoutBuiltCbk(subProb, objFuncTmp)
	# create constraints for subprob
	ineqFun = x2[1] - 1.0
	ineqFun = paramFuncSymEngine(ineqFun, x2, x1)
	addInequalityWithoutBuiltCbk(subProb, ineqFun)
	# append subprob to root prob
	subProb = stochProg(subProb)
	addScenarioStochProg(myStochProg, 0.5, subProb)
	addScenarioStochProg(myStochProg, 0.5, subProb)	
	# eps
	epsilonIter = 0.00001
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# check obj
	correctOptObj = -2.0
	@test norm(optObj - correctOptObj) <= 0.0001
	# check sol
	@test norm(finalPrimalSol - [1.0]) <= 0.0001
end

###############################################################
# now we just stack problems large number of problems independtly in the second stage
# 
# the root problem is: 
# 
#   min -x + \sum p_i Q_i(x)
#   s.t. x >= 0.0 and x <= 1.0
#
# the subproblem is:
# 
#   Q_i(x) = min {-y s.t. y \in [0, vec[i]]}
#
# opt val: can be calculated by hand
# stochastic sol: [1.0]
###############################################################

function test03()
	# total scenarios
	totScenarios = 300
	# vector of bounds
	vec = rand(totScenarios)
	# vector of probabilities
	probVec = rand(totScenarios)
	probVec = probVec / sum(probVec)
	# build main prob
	name_ = "mainProbFirstStage"
	totFirstStageDecision = 1
	totalParameters_ = 0
	mainProbFirstStage = convexParametricOptProblem(
		name_, totFirstStageDecision, totalParameters_)
	# variables
	x = convert(Array{SymEngine.Basic, 1}, [symbols("x_$i") for i in 1:totFirstStageDecision])
	p = convert(Array{SymEngine.Basic, 1}, [symbols("p_$i") for i in 1:totalParameters_])
	# obj first stage prob: -x_1
	objFuncTmp = -x[1]
	objFuncTmp = paramFuncSymEngine(objFuncTmp, x, p)
	addObjFunctionWithoutBuiltCbk(mainProbFirstStage, objFuncTmp)
	# create constraint: x_1 <= 1.0
	ineqFun = x[1] - 1.0
	ineqFun = paramFuncSymEngine(ineqFun, x, p)
	addInequalityWithoutBuiltCbk(mainProbFirstStage, ineqFun)
	# create stoch prog
	myStochProg = stochProg(mainProbFirstStage)
	# create subproblem
	for i = 1:1:totScenarios
		nameSubProb = string("subProb", i)
		totSecondStageDecisions = 1
		subProb = convexParametricOptProblem(
			nameSubProb, totSecondStageDecisions, totFirstStageDecision)
		# silent mode
		if i != 1
			subProb.silentMode = true
		end
		# create vars for sub prob
		x2 = [symbols("x_$i") for i in 1:totSecondStageDecisions]
		x1 = [symbols("p_$i") for i in 1:totFirstStageDecision]
		# create objective for subprob
		objFuncTmp = -x2[1]
		objFuncTmp = paramFuncSymEngine(objFuncTmp, x2, x1)
		addObjFunctionWithoutBuiltCbk(subProb, objFuncTmp)
		# create constraints for subprob
		ineqFun = x2[1] - vec[i]
		ineqFun = paramFuncSymEngine(ineqFun, x2, x1)
		addInequalityWithoutBuiltCbk(subProb, ineqFun)
		# append subprob to root prob
		subProb = stochProg(subProb)
		addScenarioStochProg(myStochProg, probVec[i], subProb)
	end
	# eps
	epsilonIter = 0.00001
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# check obj
	correctOptObj = -1.0 - probVec'vec
	@test norm(optObj - correctOptObj) <= 0.0001
	# check sol
	@test norm(finalPrimalSol - [1.0]) <= 0.0001
	# print
	println()
	println("correctOptObj: ", correctOptObj)
	println("optObj ", optObj)
	println("finalPrimalSol: ", finalPrimalSol)
end

###############################################################
# easy multi-stage problem
# 
# the root problem is: 
# 
#   min -x + 0.5*Q_1(x) + 0.5*Q_2(x)
#   s.t. x >= 0.0 and x <= 1.0
#
# the subproblem is:
# 
#   Q_i(x) = min {-y + 0.5*P_1(y) + 0.5*P_i(y) s.t. y \in [0, 1]}
#   P_i(x) = min {-z s.t. z \in [0, 1]}
#
# opt val: -3.0
# stochastic sol: [1.0]
###############################################################

function test04()
	# build main prob
	name_ = "mainProbFirstStage"
	totFirstStageDecision = 1
	totalParameters_ = 0
	mainProbFirstStage = convexParametricOptProblem(
		name_, totFirstStageDecision, totalParameters_)
	# variables
	x = convert(Array{SymEngine.Basic, 1}, [symbols("x_$i") for i in 1:totFirstStageDecision])
	p = convert(Array{SymEngine.Basic, 1}, [symbols("p_$i") for i in 1:totalParameters_])
	# obj first stage prob: -x_1
	objFuncTmp = -x[1]
	objFuncTmp = paramFuncSymEngine(objFuncTmp, x, p)
	addObjFunctionWithoutBuiltCbk(mainProbFirstStage, objFuncTmp)
	# create constraint: x_1 <= 1.0
	ineqFun = x[1] - 1.0
	ineqFun = paramFuncSymEngine(ineqFun, x, p)
	addInequalityWithoutBuiltCbk(mainProbFirstStage, ineqFun)
	# create stoch prog
	myStochProg = stochProg(mainProbFirstStage)
	# create subproblem
	nameSubProb = "subProb"
	totSecondStageDecisions = 1
	subProb = convexParametricOptProblem(
		nameSubProb, totSecondStageDecisions, totFirstStageDecision)
	# create vars for sub prob
	x2 = [symbols("x_$i") for i in 1:totSecondStageDecisions]
	x1 = [symbols("p_$i") for i in 1:totFirstStageDecision]
	# create objective for subprob
	objFuncTmp = -x2[1]
	objFuncTmp = paramFuncSymEngine(objFuncTmp, x2, x1)
	addObjFunctionWithoutBuiltCbk(subProb, objFuncTmp)
	# create constraints for subprob
	ineqFun = x2[1] - 1.0
	ineqFun = paramFuncSymEngine(ineqFun, x2, x1)
	addInequalityWithoutBuiltCbk(subProb, ineqFun)
	# append subprob to root prob
	subProbStochProgWithoutScen = stochProg(subProb)
	subProbStochProg = stochProg(subProb)
	addScenarioStochProg(subProbStochProg, 0.5, subProbStochProgWithoutScen)
	addScenarioStochProg(subProbStochProg, 0.5, subProbStochProgWithoutScen)
	addScenarioStochProg(myStochProg, 0.5, subProbStochProg)
	addScenarioStochProg(myStochProg, 0.5, subProbStochProg)
	# eps
	epsilonIter = 0.00001
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# test obj sol
	correctOptObj = -3.0
	@test norm(optObj - correctOptObj) <= 0.0001
	# check sol
	@test norm(finalPrimalSol - [1.0]) <= 0.0001
end

test01()
test02()
test03()
test04()
