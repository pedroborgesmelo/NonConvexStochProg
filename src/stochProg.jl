# author: pedro.borges.melo@gmail.com
# date: April/2019

##########################################################################################
# 1) 
##########################################################################################

using Logging

type stochProg
	mainProb::convexParametricOptProblem
	probScenVec::Vector{Float64}
	subProbScenVec::Vector{Any}
	# constructor
	function stochProg(mainProb_)
		probScenVec_ = Vector{Float64}()
		subProbScenVec_ = Vector{stochProg}()
		# return
		return new(mainProb_, probScenVec_, subProbScenVec_)
	end
end

function addScenarioStochProg(stochProgInst, probScen, subProbScen)
	push!(stochProgInst.probScenVec, probScen)
	push!(stochProgInst.subProbScenVec, subProbScen)
end

function getReducedFormOfObjectivesOfSubproblems(stochProgInst, epsilonIter, 
		stage, scen, prefixStr, MAX_ITER_PER_SUBPROBLEM = 100)
	# take total scenarios
	totScen = length(stochProgInst.probScenVec)
	totDecisionsLeaf = stochProgInst.mainProb.totalDecisionVariables
	totParamsLeaf = stochProgInst.mainProb.totalParameters
	# init
	totalObjectiveParamCbk = nothing
	# init vectors for weigted sum
	paramCbksForSum = Vector{parametricCallback}()
	weightsForSum = Vector{Float64}()
	# walk tree
	for i = 1:1:totScen
		# calc reduced form for the leafs
		stochProgIter = stochProgInst.subProbScenVec[i]
		prefixStrIter = string(prefixStr, "-stage-", stage, "-scen-", scen)
		reducedObjAtmCbk = getReducedFormOfObjectivesOfSubproblems(
			stochProgIter, epsilonIter, stage+1, i, prefixStrIter,
			MAX_ITER_PER_SUBPROBLEM)
		# dive reducedObjAtmCbk into a parametric callback
		reducedObjParamCbk = createParamCbkWithZerosInParamVars(
			totParamsLeaf, reducedObjAtmCbk)
		# add
		push!(paramCbksForSum, reducedObjParamCbk)
		push!(weightsForSum, stochProgInst.probScenVec[i])
	end
	# main term
	push!(paramCbksForSum, stochProgInst.mainProb.objCbk)
	push!(weightsForSum, 1.0)
	# generate objective
	totalObjectiveCbk = generateWeigthedSumOfCbk(paramCbksForSum, weightsForSum)	
	# final part
	if totParamsLeaf > 0
		# create another optimization problem for the total reduced objective
		name_ = string(prefixStr, "-stage-", stage, "-scen-", scen)
		partialProbIter = convexParametricOptProblem(
			name_, totDecisionsLeaf, totParamsLeaf)
		# prob type
		partialProbIter.probType = stochProgInst.mainProb.probType
		# silent mode
		partialProbIter.silentMode = true
		# copy constraints of the main problem of the stoch prog
		takeConstraintsFromToConvexProb(
			stochProgInst.mainProb, partialProbIter)
		# set objective of total reduced problem
		addObjFunctionCbk(partialProbIter, totalObjectiveCbk)		
		# take smoothed final objective
		smoothAproxOptVal = getCbkOfSmoothApproximationToOptimalValue(
			partialProbIter, epsilonIter, 
			MAX_ITER_PER_SUBPROBLEM)
		# return
		return smoothAproxOptVal
	else
		# return
		return totalObjectiveCbk
	end
end

function buildSolvableStochProg(stochProgRoot, 
		epsilonIter, MAX_ITER_PER_SUBPROBLEM = 100)
	# create new optimization problem
	name_ = "decomposableEquivalent"
	totalDecisionVariables_ = stochProgRoot.mainProb.totalDecisionVariables
	totalParameters_ = stochProgRoot.mainProb.totalParameters
	decomposableEquivalent = convexParametricOptProblem(
		name_, totalDecisionVariables_, totalParameters_)
	# copy constraints of my problem in the root of the tree to final opt problem
	takeConstraintsFromToConvexProb(stochProgRoot.mainProb, decomposableEquivalent)
	# calculate final objective in reduced form
	reducedObjFormCbk = getReducedFormOfObjectivesOfSubproblems(
		stochProgRoot, epsilonIter, 1, 0, "", MAX_ITER_PER_SUBPROBLEM)
	# set objective of final opt problem
	addObjFunctionCbk(decomposableEquivalent, reducedObjFormCbk)
	# return
	return decomposableEquivalent
end

function solveBuiltStochProg(decomposableEquivalent::convexParametricOptProblem, 
		initialIterate::Array{Float64, 1}, 
		hessianFlag = false, MAX_ITER_PER_SUBPROBLEM = 2000)
	# null param
	totalParameters_ = decomposableEquivalent.totalParameters
	paramIter = rand(totalParameters_)
	# solve
	optObj, finalPrimalSol = calcOptProblemSolution(
		decomposableEquivalent, paramIter, initialIterate, 
		hessianFlag, MAX_ITER_PER_SUBPROBLEM)
	# return
	return optObj, finalPrimalSol
end
