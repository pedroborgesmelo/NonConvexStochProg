# author: pedro.borges.melo@gmail.com
# date: April/2019

using Base.Test

include("../src/stochProgUpperSmoothing.jl")

###############################################################
# create a standard linear prog with no scenarios
# 
# the root problem is:
# 
#   min -x
#   s.t. x >= 0.0 and x <= 1.0
# 
# have to make a transformation to fit the standard form:
#
#   min -x
#   s.t. x >= 0, s >= 0
#        1.0 - x - s = 0
# 
# sol is: [x, s] = [1, 0]
# opt obj is: -1.0
###############################################################

function test01()
	# create std linear prog
	totDecisionVars_ = 2
	totEqualityConstraints_ = 1
	costQ_ = [-1.0, 0]
	rhsH_ = [1.0]
	matrixW_ = zeros(totEqualityConstraints_, totDecisionVars_)
	matrixW_[1, :] = [1.0, 1.0]
	matrixT_ = spzeros(totEqualityConstraints_, 0)
	myRhsParamLinearStochProg_ = rhsParametricLinearProb(costQ_, rhsH_, matrixW_, matrixT_)
	# build stoch prog
	myStdLinearStochProg = stdLinearStochProg(myRhsParamLinearStochProg_)
	myStochProg = buildStochProgFromStdStochLinearProg(myStdLinearStochProg)
	# eps
	epsilonIter = 0.1
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0, 0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# print
	println()
	println("optObj: ", optObj)
	println("finalPrimalSol: ", finalPrimalSol)
	# test obj
	correctOptObj = -1.0
	@test norm(optObj - correctOptObj) <= 0.001
	# test sol
	correctOptSol = [1.0, 0.0]
	@test norm(finalPrimalSol - correctOptSol) <= 0.001
end

###############################################################
# create a standard linear prog with no scenarios
# 
# the root problem is:
# 
#   min -x + 0.5*Q_1(x) + 0.5*Q_2(x)
#   s.t. x >= 0.0 and x <= 1.0
#
# where:
#   
#   Q_i(x) = min {-y s.t. y \in [0,1]}
# 
# have to make a transformation to fit the standard form:
#
#   min -x
#   s.t. x >= 0, s >= 0
#        1.0 - x - s = 0
# 
# sol is: [x, s] = [1, 0]
# opt obj is: -1.0
###############################################################

function test02()
	# create std linear prog
	totDecisionVars_ = 2
	totEqualityConstraints_ = 1
	costQ_ = [-1.0, 0]
	rhsH_ = [1.0]
	matrixW_ = zeros(totEqualityConstraints_, totDecisionVars_)
	matrixW_[1, :] = [1.0, 1.0]
	matrixT_ = spzeros(totEqualityConstraints_, 0)
	myRhsParamLinearStochProg_ = rhsParametricLinearProb(costQ_, rhsH_, matrixW_, matrixT_)
	# build root stoch prog
	myStdLinearStochProg = stdLinearStochProg(myRhsParamLinearStochProg_)
	# scenario sub problem
	totDecisionVars_ = 2
	totEqualityConstraints_ = 1
	costQ_ = [-1.0, 0]
	rhsH_ = [1.0]
	matrixW_ = zeros(totEqualityConstraints_, totDecisionVars_)
	matrixW_[1, :] = [1.0, 1.0]
	matrixT_ = spzeros(totEqualityConstraints_, 2)
	myRhsParamLinearStochProgSubprob_ = rhsParametricLinearProb(costQ_, rhsH_, matrixW_, matrixT_)
	scenCopyStdLinearStochProg = stdLinearStochProg(myRhsParamLinearStochProgSubprob_)	
	# add scenarios
	addSubProblemStdLinearStochProg(myStdLinearStochProg, scenCopyStdLinearStochProg, 0.5)
	addSubProblemStdLinearStochProg(myStdLinearStochProg, scenCopyStdLinearStochProg, 0.5)
	# built general stoch prog
	myStochProg = buildStochProgFromStdStochLinearProg(myStdLinearStochProg)
	# eps
	epsilonIter = 0.00001
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0, 0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# print
	println()
	println("optObj: ", optObj)
	println("finalPrimalSol: ", finalPrimalSol)
	# test obj
	correctOptObj = -2.0
	@test norm(optObj - correctOptObj) <= 0.001
	# test sol
	correctOptSol = [1.0, 0.0]
	@test norm(finalPrimalSol - correctOptSol) <= 0.001
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
	totScenarios = 800
	# vector of bounds
	vec = rand(totScenarios)
	# vector of probabilities
	probVec = rand(totScenarios)
	probVec = probVec / sum(probVec)
	# create std linear prog
	totDecisionVars_ = 2
	totEqualityConstraints_ = 1
	costQ_ = [-1.0, 0]
	rhsH_ = [1.0]
	matrixW_ = zeros(totEqualityConstraints_, totDecisionVars_)
	matrixW_[1, :] = [1.0, 1.0]
	matrixT_ = spzeros(totEqualityConstraints_, 0)
	myRhsParamLinearStochProg_ = rhsParametricLinearProb(costQ_, rhsH_, matrixW_, matrixT_)
	# build root stoch prog
	myStdLinearStochProg = stdLinearStochProg(myRhsParamLinearStochProg_)
	# built problems for scenarios
	for i = 1:1:totScenarios
		totDecisionVars_ = 2
		totEqualityConstraints_ = 1
		costQ_ = [-1.0, 0]
		rhsH_ = [vec[i]]
		matrixW_ = zeros(totEqualityConstraints_, totDecisionVars_)
		matrixW_[1, :] = [1.0, 1.0]
		matrixT_ = spzeros(totEqualityConstraints_, 2)
		myRhsParamLinearStochProgSubprob_ = rhsParametricLinearProb(costQ_, rhsH_, matrixW_, matrixT_)
		scenCopyStdLinearStochProg = stdLinearStochProg(myRhsParamLinearStochProgSubprob_)	
		# add scenarios
		addSubProblemStdLinearStochProg(myStdLinearStochProg, scenCopyStdLinearStochProg, probVec[i])
	end
	# built general stoch prog
	myStochProg = buildStochProgFromStdStochLinearProg(myStdLinearStochProg)
	# eps
	epsilonIter = 0.00001
	# build solvable stoch prog
	decomposableEquivalent = buildSolvableStochProg(myStochProg, epsilonIter)
	# solve
	initialIterate = [0.0, 0.0]
	optObj, finalPrimalSol = solveBuiltStochProg(decomposableEquivalent, initialIterate)
	# print
	println()
	println("optObj: ", optObj)
	println("finalPrimalSol: ", finalPrimalSol)
	println()
	# check obj
	correctOptObj = -1.0 - probVec'vec
	@test norm(optObj - correctOptObj) <= 0.0001
	# print obj
	println()
	println("correctOptObj: ", correctOptObj)
	println("optObj: ", optObj)
	println()
	# test sol
	correctOptSol = [1.0, 0.0]
	@test norm(finalPrimalSol - correctOptSol) <= 0.001
end

test01()
test02()
test03()
