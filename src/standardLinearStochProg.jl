# author: pedro.borges.melo@gmail.com
# date: April/2019

##########################################################################################
# 0) rhsParametricLinearProb is for the sens. wrt to x of min {q'y : y >=0, Wy = h - Tx}
# 1) data structures here are filled and the linear parametric callbacks are built
# 2) note that the parametric analysis is only wrt to the right hand side
##########################################################################################

using Logging

type rhsParametricLinearProb
	costQ::SparseVector{Float64,Int64}
	rhsH::SparseVector{Float64,Int64}
	matrixW::SparseMatrixCSC{Float64,Int64}
	matrixT::SparseMatrixCSC{Float64,Int64}
	# constructor
	function rhsParametricLinearProb(costQ_, rhsH_, matrixW_, matrixT_)
		# return
		return new(costQ_, rhsH_, matrixW_, matrixT_)
	end
end

type stdLinearStochProg
	mainProb::rhsParametricLinearProb
	# scanario probs
	scenarioProbVec::Vector{Any}
	probVec::Vector{Float64}
	# constructor
	function stdLinearStochProg(rhsParametricLinearProb_)
		scenarioProbVec_ = Vector{stdLinearStochProg}()
		probVec_ = Vector{Float64}()
		# return
		return new(rhsParametricLinearProb_, scenarioProbVec_, probVec_)
	end
end

function getTotalFirstStageDecisionsStdLinearStochProg(stdLinearStochProg)
	return getTotalDecisionVarsRhsParametricLinearProb(stdLinearStochProg.mainProb)
end

function getTotalDecisionVarsRhsParametricLinearProb(rhsParametricLinearProb)
	return length(rhsParametricLinearProb.costQ)
end

function getTotalEqualityConstraintsRhsParametricLinearProb(rhsParametricLinearProb)
	return length(rhsParametricLinearProb.rhsH)
end

function getTotalParamsRhsParametricLinearProb(rhsParametricLinearProb)
	return size(rhsParametricLinearProb.matrixT, 2)
end

function getTotalParamsStdLinearStochProg(stdLinearStochProg)
	return getTotalParamsRhsParametricLinearProb(
		stdLinearStochProg.mainProb)
end

function getTotalScenariosStdLinearStochProg(stdLinearStochProg)
	return length(stdLinearStochProg.scenarioProbVec)
end

function addSubProblemStdLinearStochProg(stdLinearStochProg, subProb, prob)
	push!(stdLinearStochProg.scenarioProbVec, subProb)
	push!(stdLinearStochProg.probVec, prob)
end

function createAtmCbkForFreeTermOfEqualityConstriantsRhsParametricLinearProb(
		matrixT::SparseMatrixCSC{Float64,Int64},
		rhsH::SparseVector{Float64,Int64},
		equalityConstrIdx::Int64)
	# dimensions
	totalParams_ = size(matrixT, 2)
	# entry size
	entrySize_ = totalParams_
	# value cbk
	function valueCbk_(z::Array{Float64, 1})
		tmp = rhsH[equalityConstrIdx]
		tmp -= matrixT[equalityConstrIdx,:]'z
		# return
		return tmp
	end
	# grad cbk
	function gradCbk_(grad::Array{Float64, 1}, z::Array{Float64, 1})
		sizeZ = length(z)
		for i = 1:1:sizeZ
			grad[i] = - matrixT[equalityConstrIdx,i]
		end
	end
	# hess cbk 
	function hessCbk_(result::SparseMatrixCSC{Float64,Int64}, z::Array{Float64, 1})
		# NOTE: nothing to be done
		# NOTE: result is zero
	end
	# third order tensor cbk 
	function thirdOrderTensorCbk_(result::mySparseTensor, z::Array{Float64, 1})
		# NOTE: nothing to be done
		# NOTE: result is zero
	end
	# create atomic callback
	atmCbk_ = atomicCallback(entrySize_, valueCbk_, gradCbk_, hessCbk_, thirdOrderTensorCbk_)
	# return
	return atmCbk_
end

function createMtxCbkCbkForEqConstrFromRhsParametricLinearProb(
		rhsParametricLinearProb)
	# get dimensions
	totParamsRhsLinParamProb = getTotalParamsRhsParametricLinearProb(rhsParametricLinearProb)
	totDecisionVars = getTotalDecisionVarsRhsParametricLinearProb(rhsParametricLinearProb)
	totEqConstraints = getTotalEqualityConstraintsRhsParametricLinearProb(rhsParametricLinearProb)
	# create main matrix eq constraints
	entrySize_ = totParamsRhsLinParamProb
	numLines_ = totEqConstraints
	numCols_ = totDecisionVars
	eqConstrMtxCbk = matrixCallback(entrySize_, numLines_, numCols_)
	eqConstrMtxCbk.nonParametricPartMtxCbk = rhsParametricLinearProb.matrixW
	# init rhs matrix eq constraints
	entrySize_ = totParamsRhsLinParamProb
	numLines_ = totEqConstraints
	numCols_ = 1
	rhsMtxCbk = matrixCallback(entrySize_, numLines_, numCols_)
	# iterate eq constraints
	for i = 1:1:totEqConstraints
		# gen free term
		freeTermIter = createAtmCbkForFreeTermOfEqualityConstriantsRhsParametricLinearProb(
			rhsParametricLinearProb.matrixT, rhsParametricLinearProb.rhsH, i)
		setMtxCbkPos(rhsMtxCbk, i, 1, freeTermIter)
	end
	# return
	return eqConstrMtxCbk, rhsMtxCbk
end

# NOTE: problem is linear, but the recourse terms will be general param cbks
# NOTE: therefore, here we create param cbks also
function createParamCbkForObjCbkFromRhsParametricLinearProb(costQ, totalParams_)
	# dimensions
	totalDecisionVars_ = length(costQ)
	# entry size
	entrySize_ = totalDecisionVars_ + totalParams_
	# value cbk
	function valueCbk_(z::Array{Float64, 1})
		x = z[1:totalDecisionVars_]
		costTmpQ = costQ
		# return
		return costTmpQ'x
	end
	# grad cbk
	function gradCbk_(grad::Array{Float64, 1}, z::Array{Float64, 1})
		x = z[1:totalDecisionVars_]
		# fill grad in x
		for i = 1:1:totalDecisionVars_
			grad[i] = costQ[i]
		end
		# fill grad in p
		for i = 1:1:totalParams_
			grad[i+totalDecisionVars_] = 0.0
		end
	end
	# hess cbk 
	function hessCbk_(result::SparseMatrixCSC{Float64,Int64}, z::Array{Float64, 1})
		# NOTE: nothing to be done
		# NOTE: result is zero
	end
	# third order tensor cbk 
	function thirdOrderTensorCbk_(result::mySparseTensor, z::Array{Float64, 1})
		# NOTE: nothing to be done
		# NOTE: result is zero
	end
	# create atomic callback for the base of parametric callback
	atmCbk_ = atomicCallback(entrySize_, valueCbk_, gradCbk_, hessCbk_, thirdOrderTensorCbk_)
	# param cbk
	paramCbkObj = parametricCallback(totalDecisionVars_, totalParams_, atmCbk_)
	# return
	return paramCbkObj
end

function buildStochProgFromStdStochLinearProgRecursion(stdLinearStochProg, stage::Int64, scen::Int64)
	# get main prob
	rhsParametricLinearProb = stdLinearStochProg.mainProb
	# setup
	name_ = string("prob-stage-", stage, "-scen-", scen)
	totFirstStageDecision = getTotalDecisionVarsRhsParametricLinearProb(rhsParametricLinearProb)
	totalParameters_ = getTotalParamsRhsParametricLinearProb(rhsParametricLinearProb)
	# create main prob
	mainProbFirstStage = convexParametricOptProblem(
		name_, totFirstStageDecision, totalParameters_)
	# create matrix callback
	eqConstrMtxCbk, rhsMtxCbk = createMtxCbkCbkForEqConstrFromRhsParametricLinearProb(rhsParametricLinearProb)
	mainProbFirstStage.mtxCbk = eqConstrMtxCbk
	mainProbFirstStage.rhsCbk = rhsMtxCbk
	# create linear parametric obj
	paramObjCbk = createParamCbkForObjCbkFromRhsParametricLinearProb(
		rhsParametricLinearProb.costQ, totalParameters_)
	# add linear parametric obj
	addObjFunctionCbk(mainProbFirstStage, paramObjCbk)
	# create main stoch prog
	finalStochProg = stochProg(mainProbFirstStage)
	# walk tree
	totalScenarios = length(stdLinearStochProg.probVec)
	for i = 1:1:totalScenarios
		stdLinearStochProgScen = stdLinearStochProg.scenarioProbVec[i]
		scenStochProg = buildStochProgFromStdStochLinearProgRecursion(
			stdLinearStochProgScen, stage+1, i)
		# add scenario problem
		prob = stdLinearStochProg.probVec[i]
		addScenarioStochProg(finalStochProg, prob, scenStochProg)
	end
	# problems are fully linear at the leafs
	# NOTE: this is a very bad trick to avoid filling hessian in the parametric analysis engine
	if totalScenarios == 0
		finalStochProg.mainProb.probType = "linear"
	end
	# return
	return finalStochProg
end

function buildStochProgFromStdStochLinearProg(stdLinearStochProg)
	# start recursion
	finalStochProg = buildStochProgFromStdStochLinearProgRecursion(stdLinearStochProg, 1, 0)
	# return
	return finalStochProg
end

function buildSpecialMatrixMapp(rhsH, matrixT)
	# init
	entrySize_ = size(matrixT, 2)
	numLines_ = size(matrixT, 1)
	numCols_ = 1
	# init callback
	rhsMtxCbk = matrixCallback(entrySize_, numLines_, numCols_)
	# fill lines
	for i = 1:1:numLines_
		freeTermIter = createAtmCbkForFreeTermOfEqualityConstriantsRhsParametricLinearProb(
			matrixT, rhsH, i)
		setMtxCbkPos(rhsMtxCbk, i, 1, freeTermIter)
	end
	# return
	return rhsMtxCbk
end

function createConvexParametricOptProblemFromRhsParametricLinearProbWithSquareDependence(
		rhsParametricLinearProb, stage::Int64, scen::Int64,
		squareMatrix::SparseMatrixCSC{Float64,Int64},
		constantVec::SparseVector{Float64,Int64})
	# setup
	name_ = string("prob-stage-", stage, "-scen-", scen)
	totFirstStageDecision = getTotalDecisionVarsRhsParametricLinearProb(rhsParametricLinearProb)
	totalParameters_ = size(squareMatrix, 2)
	# create main prob
	convexParamProb = convexParametricOptProblem(
		name_, totFirstStageDecision, totalParameters_)
	# get dimensions
	totParamsRhsLinParamProb = getTotalParamsRhsParametricLinearProb(rhsParametricLinearProb)
	totDecisionVars = getTotalDecisionVarsRhsParametricLinearProb(rhsParametricLinearProb)
	totEqConstraints = getTotalEqualityConstraintsRhsParametricLinearProb(rhsParametricLinearProb)
	# create matrix callback for equality constraints
	entrySize_ = totParamsRhsLinParamProb
	numLines_ = totEqConstraints
	numCols_ = totDecisionVars
	eqConstrMtxCbk = matrixCallback(entrySize_, numLines_, numCols_)
	eqConstrMtxCbk.nonParametricPartMtxCbk = rhsParametricLinearProb.matrixW
	# create the rhs matrix callback
	# NOTE: the difference here is that the rhs is the identity map
	# NOTE: this improves computational performance provided 
	# NOTE: that number of variables is larger than number of constraints
	entrySize_ = totParamsRhsLinParamProb
	numLines_ = totEqConstraints
	numCols_ = 1
	rhsMtxCbk = buildSpecialMatrixMapp(constantVec, squareMatrix)
	# add equality constraints to convex parametric problem
	convexParamProb.mtxCbk = eqConstrMtxCbk
	convexParamProb.rhsCbk = rhsMtxCbk
	# create linear parametric obj
	paramObjCbk = createParamCbkForObjCbkFromRhsParametricLinearProb(
		rhsParametricLinearProb.costQ, totalParameters_)
	# add linear parametric obj
	addObjFunctionCbk(convexParamProb, paramObjCbk)
	# return
	return convexParamProb
end

function createConvexParametricOptProblemFromRhsParametricLinearProbWithoutDependenceOnMatrixT(
		rhsParametricLinearProb, stage::Int64, scen::Int64)
	# params
	totEqConstraints = getTotalEqualityConstraintsRhsParametricLinearProb(rhsParametricLinearProb)
	idMatrix = -1.0 .* speye(totEqConstraints)
	zeroVec = spzeros(totEqConstraints)
	# delegate
	return createConvexParametricOptProblemFromRhsParametricLinearProbWithSquareDependence(
		rhsParametricLinearProb, stage, scen, idMatrix, zeroVec)
end

function calcReducedProblemUsingEfficientCompositionRecursion(partialProbIter, stdLinearStochProg, 
		epsilonIter::Float64, MAX_ITER_PER_SUBPROBLEM::Int64, stage::Int64, scen::Int64)
	# split in cases for efficiency
	totalScenarios = length(stdLinearStochProg.probVec)
	totalParameters_ = getTotalParamsStdLinearStochProg(stdLinearStochProg)
	if totalScenarios > 0
		# init vectors for weigted sum
		paramCbksForSum = Vector{parametricCallback}()
		weightsForSum = Vector{Float64}()
		# iterate scenarios
		for i = 1:1:totalScenarios
			# get scenario problem
			stdLinearStochProgScen = stdLinearStochProg.scenarioProbVec[i]
			# create scenario problem
			convParamProbScen = createConvexParametricOptProblemFromRhsParametricLinearProbWithoutDependenceOnMatrixT(
				stdLinearStochProgScen.mainProb, stage+1, i)
			# silent mode
			convParamProbScen.silentMode = true
			# recursion step
			convParamProbScen = calcReducedProblemUsingEfficientCompositionRecursion(
				convParamProbScen, stdLinearStochProgScen, epsilonIter, MAX_ITER_PER_SUBPROBLEM, stage+1, i)
			# get smooth approx to opt value function depending only on the right hand side without rhsH and matrixT
			smoothAproxOptValAtmCbk = getCbkOfSmoothApproximationToOptimalValue(
				convParamProbScen, epsilonIter, MAX_ITER_PER_SUBPROBLEM)
			# compose with matrix mapp efficiently
			rhsParamLinProb = stdLinearStochProgScen.mainProb
			smoothAproxOptValWithEfficientComposositionAtmCbk = composeAtomicCallbackWithAffineMatrixMapp(
				smoothAproxOptValAtmCbk, rhsParamLinProb.rhsH, rhsParamLinProb.matrixT)
			# create param cbk: dive reducedObjAtmCbk into a parametric callback
			smoothAproxOptValParamCbk = createParamCbkWithZerosInParamVars(
				totalParameters_, smoothAproxOptValWithEfficientComposositionAtmCbk)
			# push to vector to be summed
			push!(paramCbksForSum, smoothAproxOptValParamCbk)
			push!(weightsForSum, stdLinearStochProg.probVec[i])
		end
		# add linear cost to vector to be summed
		push!(paramCbksForSum, partialProbIter.objCbk)
		push!(weightsForSum, 1.0)
		# calculate weighted sum
		summedAndWeidghtedParamCbk = generateWeigthedSumOfCbk(paramCbksForSum, weightsForSum)
		# set final objective
		partialProbIter.objCbk = summedAndWeidghtedParamCbk
	else
		partialProbIter.probType = "linear"
	end
	# return
	return partialProbIter
end

function buildSolvableProgFromStdStochLinearProg(stdLinearStochProg, 
		epsilonIter, MAX_ITER_PER_SUBPROBLEM)
	# init
	stage = 1
	scen = 1
	# create first stage problem
	rhsParametricLinearProb = stdLinearStochProg.mainProb
	totEqConstraints = getTotalEqualityConstraintsRhsParametricLinearProb(rhsParametricLinearProb)
	partialProbIter = createConvexParametricOptProblemFromRhsParametricLinearProbWithSquareDependence(
		rhsParametricLinearProb, stage, scen, rhsParametricLinearProb.matrixT, rhsParametricLinearProb.rhsH)
	# create root optimization problem
	decomposableEquivalent = calcReducedProblemUsingEfficientCompositionRecursion(
		partialProbIter, stdLinearStochProg, epsilonIter, MAX_ITER_PER_SUBPROBLEM, stage, scen)
	# return
	return decomposableEquivalent
end

function sampleMatrixWithSparsityIndex(numRows, numCols, sparsityIndex)
	# calc total non-zeros positions
	totalNonZeroPositions = 1 + round(numRows*numCols*sparsityIndex)
	totalNonZeroPositions = min(numRows*numCols, totalNonZeroPositions)
	# sample until total non-zero positions is filled
	randomMatrix = spzeros(numRows, numCols)
	nonZeroPositions = Set{Array{Int64, 1}}()
	if numRows > 0 && numCols > 0
		while length(nonZeroPositions) < totalNonZeroPositions
			randRow = rand(1:numRows)
			randCol = rand(1:numCols)
			pos = [randRow, randCol]
			push!(nonZeroPositions, pos)
			randomMatrix[randRow, randCol] = rand()
		end
	end
	# return
	return randomMatrix
end

function generateRandomStdLinearStochProgRecursion(totalStages, 
		totalScenariosPerNode, dimNonSlackDecisionVarsPerStage,
		dimEqualityConstraintsPerStage, currentStage, sparsityIndex)
	# set dimensions
	numRowsW = dimEqualityConstraintsPerStage[currentStage]
	numColsW = dimNonSlackDecisionVarsPerStage[currentStage]
	numRowsT = dimEqualityConstraintsPerStage[currentStage]
	# get dimensions previous stage
	dimRecourseVar = 0
	totSlacksPreviousStage = 0
	if currentStage >= 2
		dimRecourseVar = dimNonSlackDecisionVarsPerStage[currentStage-1]
		totSlacksPreviousStage = 2*dimEqualityConstraintsPerStage[currentStage-1]
	end
	numColsT = dimRecourseVar
	# create data
	costQ_ = rand(numColsW)
	rhsH_ = rand(numRowsW)
	matrixW_ = sampleMatrixWithSparsityIndex(numRowsW, numColsW, sparsityIndex)
	matrixT_ = sampleMatrixWithSparsityIndex(numRowsT, numColsT, sparsityIndex)
	# put slack variables for feasibility
	# s+ and s- for each eq constraint
	totSlacksPerSign = dimEqualityConstraintsPerStage[currentStage]
	totSlacks = 2*totSlacksPerSign
	# slack for costQ
	withSlackCostQ_ = spzeros(numColsW + totSlacks)
	withSlackCostQ_[1:numColsW] = costQ_
	bigM = 1000.0
	withSlackCostQ_[(numColsW+1):1:end] = bigM
	# slack for W
	withSlackMatrixW_ = spzeros(numRowsW, numColsW + totSlacks)
	withSlackMatrixW_[1:numRowsW, 1:numColsW] = matrixW_
	for i = 1:1:totSlacksPerSign
		withSlackMatrixW_[i, numColsW + i] = 1.0
		withSlackMatrixW_[i, numColsW + totSlacksPerSign + i] = -1.0
	end
	# slack for T
	withSlackMatrixT_ = spzeros(numRowsT, numColsT + totSlacksPreviousStage)
	withSlackMatrixT_[1:numRowsT, 1:numColsT] = matrixT_
	# create rhs parametric linear prob
	rhsParamProg = rhsParametricLinearProb(withSlackCostQ_, rhsH_, withSlackMatrixW_, withSlackMatrixT_)
	# create std linear stoch prog
	stdLinStochProg = stdLinearStochProg(rhsParamProg)
	# recursion
	if currentStage + 1 <= length(dimNonSlackDecisionVarsPerStage)
		# total scenarios stage
		totScenariosStage = convert(Int64, totalScenariosPerNode[currentStage])
		# sample probabilities of scenarios
		probVec = rand(totScenariosStage)
		probVec = probVec / sum(probVec)
		# add scenarios
		for i = 1:1:totScenariosStage
			stdLinStochProgScen = generateRandomStdLinearStochProgRecursion(totalStages, 
				totalScenariosPerNode, dimNonSlackDecisionVarsPerStage,
				dimEqualityConstraintsPerStage, currentStage+1, sparsityIndex)
			# add scenario prob
			addSubProblemStdLinearStochProg(stdLinStochProg, stdLinStochProgScen, probVec[i])
		end
	end
	# return
	return stdLinStochProg
end

function writeRandomStdLinearStochProg(totalStages, sparsityIndex,
		totalScenariosPerNode, dimNonSlackDecisionVarsPerStage, 
		dimEqualityConstraintsPerStage, pathToSerializeProb)
	# recursion
	currentStage = 1
	randomStdLinearStochProg = generateRandomStdLinearStochProgRecursion(
		totalStages, totalScenariosPerNode, dimNonSlackDecisionVarsPerStage, 
		dimEqualityConstraintsPerStage, currentStage, sparsityIndex)
	# serialize
	fileToSerializeProb = open(pathToSerializeProb, "w")
	serialize(fileToSerializeProb, randomStdLinearStochProg)
	close(fileToSerializeProb)
end

function readStdLinearStochProgFromFile(pathSerializedProb)
	# deserialize
	file = open(pathSerializedProb, "r")
	storedStdLinearStochProg = deserialize(file)
	close(file)
	# return
	return storedStdLinearStochProg
end

function printInfoForStdLinearStochProg(stdLinearStochProg)
	println()
	println("total decision variables first stage: ", 
		getTotalFirstStageDecisionsStdLinearStochProg(stdLinearStochProg))
	println("total scenarios: ",
		getTotalScenariosStdLinearStochProg(stdLinearStochProg))
	println("total equality constraints first stage: ",
		getTotalEqualityConstraintsRhsParametricLinearProb(
			stdLinearStochProg.mainProb))
	println()
end
