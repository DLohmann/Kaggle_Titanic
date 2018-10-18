# Kaggle_Titanic
This is an entry in Kaggle's Titanic competition.


Titanic Kaggle Competition: https://www.kaggle.com/c/titanic

Logistic regression inspired by: https://www.youtube.com/watch?v=H6ii7NFdDeg

TODO:
- Finish ticketToNum and cabinToNum
- Include a way to use passenger's name in data. This could give info about whether a passenger survived. Factors such as a woman's prefix "Ms." vs "Mrs." and marriage data might potentially affect probability of survival? Maybe the type of name could influence the probability?
- Extract features like deck number
	https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
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
	
	- Figure out how to pass the (partial) derivatives of the logistic function into scipy.optimize in order allow the minimize function to use gradient-based optimization algorithms that can speed up the optimization process. Gradient-based optimization algorithms are sometimes faster. And the derivative of the logistic function is already known.

- Do the classification with the nearest neighbors algorithm. But instead of picking "k" neighbors, use all the neighbors, and assign each neighbor a weight which is the inverse of the Euclidean distance to that neighbor squared (1/r**2), and to calculate the Euclidean distance, assign each "i"th attribute in vector a weight W[i] which is chosen during optimization (so some dimensions are more important than others).

- Add more and different error functions (loss functions) besides sum squared error

- Organize files so that code for different machine learning algorithms does not mix

- Use numpy or scipy arrays and vector operations instead of regular python arrays  and custom functions to improve speed. Many of these libraries may be written in C/C++, and will run faster than vector operations written in Python

- Make the training code run in C++ for speed

- Write a script to ssh the code to Google Cloud, Amazon Web service, Azure, or UC Merced's servers to train (if the code takes to long to train).