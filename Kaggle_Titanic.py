#!/usr/bin/env python

#Python libraries
#import matplotlib.pyplot as plt
import scipy
import random as rand	#Used to determine initial weights
from scipy.optimize import minimize
import heapq	#USed to hold nearest neighbors

#My project helper files
import read_files #Reads in the data

passengerDataVectors = []
survivalVector = [] # Vector indicating the survival of passengers in the training data set


def logistic (x, derivative=False):
	if (derivative):
		exponent = scipy.exp(x)
		return exponent/((1+exponent)**2)
	return 1/(1+scipy.exp(-1*x))


def vector_multiply (A, B):
	if (len(A) != len(B)): # Error checking
		raise ValueError ("Error: can't dot 2 vectors of different lengths.")
		return None
	return sum(A[i]*B[i] for i in range (len(A)))


#W = []	#Weights for linear combination in logistic function
#def linearCombine(dataVect):
	#if (len(W) != len (dataVect) + 1):
	#	raise ValueError ("Error: weight vector and (data vector +1) lengths do not match")
	#return vector_multiply(dataVect.append(1.0))


# Multiply vector by a scalar
def vector_scalar_multiply (c, A):
	return [c * A[i] for i in range (len(A))]

def vector_length_squared (A):
	return vector_multiply(A, A) #Maybe this can be optimized using Python's "map" function

#Find the Euclidean length of a vector
def vector_length (A):
	return scipy.sqrt(vector_length_squared(A))
	
#Add 2 vectors:
def vector_add(A, B):
	if (len(A) != len(B)): # Error checking
		raise ValueError ("Error: can't add 2 vectors of different lengths.")
		return None
	return [A[i]+B[i] for i in range (len(A))]

def vector_distance_squared (A, B):
	# Maybe this can be optimized using Python's "reduce" function?
	differenceVector = vector_add(A, vector_scalar_multiply(-1.0, B))
	return vector_length_squared(differenceVector)
	
# Find the Euclidean distance between 2 vectors, A and B
def vector_distance (A, B):
	return scipy.sqrt(vector_distance_squared(A, B))

# For logistic regression
def predictLogistic (vectList, W):
	return logistic(vector_multiply (vectList, W))

# For k nearest neighbors
def predictKNearestNeighbors (vectList, W):
	K = 3
	global passengerDataVectors
	global survivalVector
	
	#Store the K nearest neighbors, sorted from furthest to nearest. 
	# Store as a tuple, with the distance and the neighbor vector
	nearestNeighbors = [(float('inf'), None) for i in range (K)] # stores the vectors of the nearest neighbors
	
	# The same index in each of these arrays corresponds to the same neighbor
	#nearestNeighbors = []
	#nearestDistances = [(float('inf'), None) for i in range (K)] # Stores the distances to the K nearest neighbors. Initialized to infinity so that no neighbor is further
	predictVect =   [vectList[j] * W[j] for j in range(len(W))]
	for i in range (len(passengerDataVectors)):
		passengerVect = [passengerDataVectors[i][j] * W[j] for j in range(len(W))]
		
		
		d2 = vector_distance_squared (predictVect, passengerVect)
		
		if (d2 < nearestNeighbors[0][0]): # If the distance is smaller than the furthest neighbor, then remove the furthest neighbor and add this neighbor
			heapq._heapreplace_max(nearestNeighbors, (d2, i))
			#heapq._heapify_max(nearestNeighbors) # Ensure that the largest element is at the front of the heap
			
	
	#for neighbor in nearestNeighbors:
	#	print(neighbor)
		
	# Compute average vote of neighbors
	return sum([survivalVector[nearestNeighbors[i][1]] for i in range(K)])/K
	
	

# For inverse squared blended nearest neighbors
def predictBlendedNearestNeighbors (vectList, W):
	global passengerDataVectors
	global survivalVector
	
	sumPrediction = 0
	
	for i in range(len(passengerDataVectors)):
		passengerVect = passengerDataVectors[i]
		d2 = vector_distance_squared (vectList, passengerVect)
		
		# if the distance to this vector is 0, then just use this passenger's survival value (data has been seen before)
		if (d2 == 0):
			return survivalVector[i]
		
		sumPrediction += vector_multiply (passengerVect, W)/d2
		'''
		for i in range (len(passengerVect)):
			sumPrediction += W[i]
		'''
	
	return sumPrediction
	#return logistic(sumPrediction)
	

def predict (vectList, W):
	#For nearest neighbors:
	return  predictKNearestNeighbors (vectList, W)
	
	#For blended nearest neighbors:
	#return predictBlendedNearestNeighbors (vectList, W)
	
	#For logistic regression:
	#return predictLogistic (vectList, W)
	#return logistic(vector_multiply (vectList, W)) 

# def error (dataVect, survivalVal):
	# return abs(predict(dataVect) - survivalVal)

def error (W):
	
	# Sum squared error:
	sum = 0
	for i in range(len(passengerDataVectors)):
		sum += (predict(passengerDataVectors[i], W) - survivalVector[i])**2
		
	return sum


# This function can only find the accuracy in predicting the training data, because Kaggle's data set did not include the survival of the passengers in the testing data
def findModelAccuracyWithTrainingData (W):
	print ("findModelAccuracy called!")
	print ("There are " + str(len(passengerDataVectors)) + " passengerDataVectors")
	numCorrect = 0
	numWrong = 0
	for i in range (len(passengerDataVectors)):
		prediction = 1 if (predict(passengerDataVectors[i], W) > 0.5) else 0
		if (prediction == survivalVector[i]):
			numCorrect += 1
		else:
			numWrong += 1
			
	return numCorrect, numWrong
	
#plotting the vector dot product value (passenger vector with weights) on x axis and survival on y axis
# def plotSurvival():
	
	# survive = []
	# notSurvive = []
	
	# for i in range len(passengerDataVectors):
		# if (survivalVector[i] == 1):
			# survive.append()
		
	
# def plotLogistic (W):

#def buildModel ():
	
	
def main():
	
	
	#Read in the training data
	print ("Calling read_files.py's readTrainingData function\n")
	global passengerDataVectors
	global survivalVector
	passengerDataVectors, survivalVector = read_files.readTrainingData()
	
	#Set initial weights, and include 1 extra space for constant term
	rand.seed(33)
	W = [(0) for i in range (len(passengerDataVectors[0]))]	#+1)] # Set weights to random real numbers between -50 and 50
	print ("Weights before training are: " + str(W))
	
	
	numCorrect, numWrong = findModelAccuracyWithTrainingData(W)
	print ("predicted " + str(numCorrect) + " correctly of " + str(numCorrect + numWrong) + " total, at " + str(int(100 * numCorrect/(numCorrect + numWrong))) + "% accuracy")
	
	print ("\n------------training------------")
	results = minimize (error, W)
	W = results.x
	print ("\n Done training!\n")
	
	numCorrect, numWrong = findModelAccuracyWithTrainingData(W)
	print ("predicted " + str(numCorrect) + " correctly of " + str(numCorrect + numWrong) + " total, at " + str(int(100 * numCorrect/(numCorrect + numWrong))) + "% accuracy")
	
	
	#for vect in passengerDataVectors:
		#vect.append(1) # Add a constant 1 term as a horizontal shift for the logistic function
		#print (vect)
		
		
	
	print ("Weights after training are: " + str(W))
	
	
	
	#Test the model:
	print ("\n-----------------------------TESTING WITH TESTING DATA!!!-----------------------------\n")
	
	passengerDataVectors = read_files.readTestingData(addConstant = True) #Read in the testing data
	survivalPredictionVector = [0] * len(passengerDataVectors) # Survival prediction of passengers in testing data
	
	for i in range (len(passengerDataVectors)):
		survivalPredictionVector[i] = 1 if (predict(passengerDataVectors[i], W) > 0.5) else 0
		
	read_files.writePredictionFile(survivalPredictionVector)
	
	
#if this file is the module being run, __name__ == '__main__' Otherwise __name__ is name of module being run
if __name__ == '__main__':
	main()

#Tests the functions in this program

'''
def runTestCode ():
	c = -2
	x = [1, 2, 3]
	y = [2, 3, 4]
	print ("c            = " + str(c))
	print ("x            = " + str(x))
	print ("y            = " + str(y))
	print ("c * x        = " + str(vector_scalar_multiply(c, x)))
	print ("x + y        = " + str(vector_add(x, y)))
	print ("x * y        = " + str(vector_multiply(x, y)))
	print ("||x||**2     = " + str(vector_length_squared(x)))
	print ("||x||        = " + str(vector_length(x)))
	print ("||x - y||**2 = " + str(vector_distance_squared(x, y)))
	print ("||x - y||    = " + str(vector_distance(x, y)))
	
	global passengerDataVectors
	
	# To test K nearestNeighbors
	passengerDataVectors = [
		[0, 1, 2, 3, 4, 5],
		[3, 3, 5, 2, 4, 7],
		[3, 6, 2, 6, 2, 8],
		[9, 4, 7, 3, 8, 3],
		[4, 6, 8, 3, 9, 0]
	]
	
	global survivalVector
	survivalVector = [0, 0, 0, 1, 1]
	
	x = [5, 7, 9, 4, 10, 1]
	
	W = [1, 1, 1, 1, 1, 1]
	
	print ("K nearest neighbor prediction 1 is: " + str(predictKNearestNeighbors (x, W)))
	W = [0, 1, 0, 0, 0, 0]
	print ("K nearest neighbor prediction 2 is: " + str(predictKNearestNeighbors (x, W)))
	
	#to test logistic:
	x = [(i - 50)/10 for i in range (100)]
	y = [logistic(i) for i in x]
	yd = [logistic (i, True) for i in x]
	plt.plot(x, y)
	plt.plot(x, yd)
	plt.show()
		
	x = []
	
'''
