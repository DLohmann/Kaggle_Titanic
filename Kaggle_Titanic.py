#!/usr/bin/env python

#Python libraries
import matplotlib.pyplot as plt
import scipy
import random as rand	#Used to determine initial weights
from scipy.optimize import minimize

#My project helper files
import read_files #Reads in the data

passengerDataVectors = []
survivalVector = []

def logistic (x, derivative=False):
	if (derivative):
		exponent = scipy.exp(x)
		return exponent/((1+exponent)**2)
	return 1/(1+scipy.exp(-1*x))

#to test logistic:
# x = [(i - 50)/10 for i in range (100)]
# y = [logistic(i) for i in x]
# yd = [logistic (i, True) for i in x]
# plt.plot(x, y)
# plt.plot(x, yd)
# plt.show()
	
#x = []

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
	

def predict (vectList, W):
	return logistic(vector_multiply (vectList, W)) 

# def error (dataVect, survivalVal):
	# return abs(predict(dataVect) - survivalVal)

def error (W):
	sum = 0
	for i in range(len(passengerDataVectors)):
		sum += (predict(passengerDataVectors[i], W) - survivalVector[i])**2
		
	return sum


def findModelAccuracy (W):
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
	
	
	numCorrect, numWrong = findModelAccuracy(W)
	print ("predicted " + str(numCorrect) + " correctly of " + str(numCorrect + numWrong) + " total, at " + str(int(100 * numCorrect/(numCorrect + numWrong))) + "% accuracy")
	
	print ("\n------------training------------")
	results = minimize (error, W)
	W = results.x
	
	
	numCorrect, numWrong = findModelAccuracy(W)
	print ("predicted " + str(numCorrect) + " correctly of " + str(numCorrect + numWrong) + " total, at " + str(int(100 * numCorrect/(numCorrect + numWrong))) + "% accuracy")
	
	
	#for vect in passengerDataVectors:
		#vect.append(1) # Add a constant 1 term as a horizontal shift for the logistic function
		#print (vect)
		
		
	
	print ("Weights after training are: " + str(W))
	
	#Read in the testing data
	
	#Test the model:
	print ("\n-----------------------------TESTING WITH TESTING DATA!!!-----------------------------\n")
	
	passengerDataVectors = read_files.readTestingData()
	numCorrect, numWrong = findModelAccuracy(W)
	print ("predicted " + str(numCorrect) + " correctly of " + str(numCorrect + numWrong) + " total, at " + str(int(100 * numCorrect/(numCorrect + numWrong))) + "% accuracy")
	
	
	
#if this file is the module being run, __name__ == '__main__' Otherwise __name__ is name of module being run
if __name__ == '__main__':
	main()
