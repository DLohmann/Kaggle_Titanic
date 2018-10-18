#!/usr/bin/env python

from csv import reader
import numpy as np	# has np.array for vectors

numTrainingPassengers = 0 # The amount of passengers in the training data set

#def nameToNum (name):

def genderToNum (sex):
	if   (sex[0] == "m" or sex[0] == "M"):
		return 1.0
	elif (sex[0] == "f" or sex[0] == "F"):
		return -1.0
	else:
		raise ValueError ("\nError: unexpected sting for gender.\n")
		return None

def ageToNum (age):
	if (age == ""):
		return 29.69911765	#In Microsoft Excel, I found that this is the average age
	return float(age)
def ticketToNum (ticket):
	return None

# The only reason that this fareToNum function is needed is that there is 1 blank entry in the testing data only. But this function is not used for the training data 
def fareToNum (fare):
	if (fare == ""): # In Microsoft Excel, I found that the average fare (of the training data) is 32.20420797 
		return 32.20420797 
	return float(fare)
	
def cabinToNum (cabin):
	return None
	
def embarkToNum (embarked):
	if (len(embarked) == 0):
		return -1.0	# I found that this is most common value in Excel. There are 168 "C"'s, 77 "Q"'s, and 644 "S"'s
	elif   (embarked[0] == "C"):
		return 1.0
	elif (embarked[0] == "Q"):
		return 0.0
	elif (embarked[0] == "S"):
		return -1.0
	else:
		raise ValueError ("\nError: unexpected sting for embark.\n")
		return None

def readTrainingData ():
	print ("readTrainingData called!!!\n")
	rf = open ("train.csv", "r")
	passengerVectors = [] # List of numerical vectors (more lists) that represent each passenger's data
	survivalVector = [] # List of 0's and 1's indicating whether the passenger at that index survived
	rf.readline() #skip over first line that has the column names
	for line in reader(rf):
		#Attributes are:
		#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
		#    ,,,,,,,,,,,
		passengerVector = [
			#float(line[0]),			# PassengerId
			#float(line[1]), 		# Survived (not allowed to use)
			float(line[2]), 		# Pclass  
									# Name (not used currently)
			genderToNum(line[4]),	# Sex
			ageToNum(line[5]),		# Age
			float(line[6]),			# SibSp
			float(line[7]),			# Parch
			#ticketToNum(line[8]),	# Ticket	IMPLEMENT ME!
			float(line[9]),			# Fare
			#cabinToNum(line[10]),	# Cabin	IMPLEMENT ME!
			embarkToNum(line[11]),	# Embarked
		]
		#passengerVector = line#line.split(",")
		passengerVector.append(1) # Add a constant 1 term as a horizontal shift for the logistic function
		passengerVectors.append (passengerVector)
		survivalVector.append (float(line[1]))
		#print (line)
		
	
	#print ("passenger string is: " + passStr + "\n")	
	rf.close
	
	global numTrainingPassengers #Get the total number of passengers in the training data set
	numTrainingPassengers = len(passengerVectors)
	
	return passengerVectors, survivalVector
	
	
def readTestingData (addConstant = False):
	print ("readTestingData called!!!\n")
	rf = open ("test.csv", "r")
	passengerVectors = [] # List of numerical vectors (more lists) that represent each passenger's data
	
	rf.readline() #skip over first line that has the column names
	for line in reader(rf):
		#Attributes are:
		#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
		#    ,,,,,,,,,,,
		passengerVector = [
			#float(line[0]),			# PassengerId
			#float(line[1]), 		# Survived (not allowed to use)
			float(line[1]), 		# Pclass  
									# Name (not used currently)
			genderToNum(line[3]),	# Sex
			ageToNum(line[4]),		# Age
			float(line[5]),			# SibSp
			float(line[6]),			# Parch
			#ticketToNum(line[7]),	# Ticket	IMPLEMENT ME!
			fareToNum(line[8]),		# Fare
			#cabinToNum(line[9]),	# Cabin	IMPLEMENT ME!
			embarkToNum(line[10]),	# Embarked
		]
		#passengerVector = line#line.split(",")
		if (addConstant):
			passengerVector.append(1) # Add a constant 1 term as a horizontal shift for the logistic function
		passengerVectors.append (passengerVector)
		#survivalVector.append (float(line[1]))
		#print (line)
		
	
	#print ("passenger string is: " + passStr + "\n")	
	rf.close()
	return passengerVectors#, survivalVector

def writePredictionFile (survivalPredictionVector):
	wf = open ("Logistic_Regression_Prediction.csv", "w")
	#print ("There are " + str(numTrainingPassengers) + " passengers in the training data set." )
	wf.write ("PassengerId,Survived\n") # Write first line
	for i in range (len(survivalPredictionVector)):
		wf.write (str(numTrainingPassengers + 1 + i) + "," + str(survivalPredictionVector[i]) + "\n")
	wf.close()
	
	