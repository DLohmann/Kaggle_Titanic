#!/usr/bin/env python

from csv import reader
#import numpy as np	# has np.array for vectors

'''

Titanic Kaggle Competition: https://www.kaggle.com/c/titanic

Logistic regression inspired by: https://www.youtube.com/watch?v=H6ii7NFdDeg

TODO:
- Finish ticketToNum and cabinToNum
- Include a way to use passenger's name in data. This could give info about whether a passenger survived. Factors such as a woman's prefix "Ms." vs "Mrs." and marriage data might potentially affect probability of survival? Maybe the type of name could influence the probability?
- Extract features like deck number
- Consider changing passengerVectors and survivalVector to numpy.arrays and setting the data type to float128 to prevent the warning "RuntimeWarning: overflow encountered in exp return 1/(1+scipy.exp(-1*x))". This happens when the weights are initially set as such:
  W = [(10*rand.random() - 5) for i in range (len(passengerDataVectors[0]))]	#+1)] # Set weights to random real numbers between -5 and 5
  
- Figure out why the size of the initial weights W affects how accurately the model predicts the training data after training
  
  
  When weights are from -50 to 50, the accuracy starts at 344/891 (38%) and goes to 344/891 (38%)
    W = [(100*rand.random() - 50) for i in range (len(passengerDataVectors[0]))]	#+1)] # Set weights to random real numbers between -50 and 50
  
  When weights are from -5 to 5, the accuracy starts at 344/891 (38%) and goes to 552/891 (61%), and there is an "overflow in exp function" error
    W = [(rand.random() - 0.5) for i in range (len(passengerDataVectors[0]))]	#+1)] # Set weights to random real numbers between -5 and 5
  
  When weights are from -0.5 to 0.5, the accuracy starts at 344/891 (38%) and goes to 717/891 (80%), and there is an "overflow in exp function" error
    W = [(rand.random() - 0.5) for i in range (len(passengerDataVectors[0]))]	#+1)] # Set weights to random real numbers between -0.50 and 0.50
	
  When weights are 0, the accuracy starts at 549/891 (61%) and goes to 717/891 (80%), and there is an "overflow in exp function" error
    W = [(0) for i in range (len(passengerDataVectors[0]))]	#+1)] # Set weights to 0

	Maybe this is because the logistic function is mostly flat for x is very large (x > 10) or very small (x < -10), and so starting with larger numbers means that there is a higher probability that initially the exponent will be further from 0, where the logistic function is flatter, and so the model learns more slowly
https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
'''

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
			float(line[0]),			# PassengerId
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
	rf.close()
	return passengerVectors, survivalVector
	
	
def readTestingData ():
	print ("readTrainingData called!!!\n")
	rf = open ("test.csv", "r")
	passengerVectors = [] # List of numerical vectors (more lists) that represent each passenger's data
	
	rf.readline() #skip over first line that has the column names
	for line in reader(rf):
		#Attributes are:
		#PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
		#    ,,,,,,,,,,,
		passengerVector = [
			float(line[0]),			# PassengerId
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
		passengerVector.append(1) # Add a constant 1 term as a horizontal shift for the logistic function
		passengerVectors.append (passengerVector)
		#survivalVector.append (float(line[1]))
		#print (line)
		
	
	#print ("passenger string is: " + passStr + "\n")	
	rf.close()
	return passengerVectors#, survivalVector