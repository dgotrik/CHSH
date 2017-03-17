#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import numba as nb

def measureClassical():
	"""
	Simulates a classical measurement (in other words, the probability
	of |00> is 1/2, the probability of |11> is 1/2. This maintains the
	perfect correlation, but with classical probability.)
	"""
	r = np.random.random_sample()
	l = np.array([0,0])
	if r<1/2:
		l[0] = -1; l[1] = -1
	elif r<=1:
		l[0] = 1; l[1] = 1
	return(l)


def measureBoth(theta1, theta2):
	"""
	Simulates the measurement in an arbitrarily oriented basis of both
	qubits of a two-qubit pair which has been prepared in the positive
	Bell state.
	Input:
	theta1: the angle of the first measurement basis with respect to the
			standard basis
	theta2: the angle of the second measurement basis

	Output:
	an array of two values from the set {-1,1} gathered randomly with
	the appropriate value based on the measurement angles
	"""
	PNegNeg = .5*(np.cos((theta1-theta2)/2)**2)
	PPosPos = PNegNeg
	PNegPos = .5*(np.sin((theta1-theta2)/2)**2)
	PPosNeg = PNegPos
	r = np.random.random_sample()
	l = np.array([0,0])
	if r<PNegNeg:
		l[0] = -1; l[1] = -1
	elif r<PNegNeg + PPosPos:
		l[0] = 1; l[1] = 1
	elif r<PNegNeg + PPosPos + PPosNeg:
		l[0] = 1; l[1] = -1
	elif r<=PNegNeg + PPosPos + PPosNeg + PNegPos:
		l[0] = -1; l[1] = 1
	return(l)

def makeData(n,theta1,theta2):
	"""
	Calls measureBoth() n times for the given angles.
	Inputs:
	n = number of iterations
	theta1 = angle of first observable
	theta2 = angle of second observable

	Output:
	array of shape [n,2] of the form (+/- 1, +/- 1)
	"""
	x = np.zeros([n,2])
	for i in range(n):
		l = measureBoth(theta1,theta2)
		x[i][0] = l[0]
		x[i][1] = l[1]
	return(x)

def makeDataC(n):
	"""
	Calls the classical measurement function.
	Output: array of shape [n,2]
	"""
	x = np.zeros([n,2])
	for i in range(n):
		l = measureClassical()
		x[i][0] = l[0]
		x[i][1] = l[1]
	return(x)

def avgDataSep(x):
	avg0 = np.sum(x[0])/len(x)
	avg1 = np.sum(x[1])/len(x)
	return(np.array([avg0,avg1]))

def avgDataTog(x):
	y = np.zeros(len(x))
	for i in range(len(x)-1):
		y[i] = x[i][0]*x[i][1]
	avg = np.sum(y)/len(y)
	return(avg)

def CHSH(n):
	a = avgDataTog(makeData(n,0,np.pi/4))
	b = avgDataTog(makeData(n,0,-np.pi/4))
	c = avgDataTog(makeData(n,np.pi/2,np.pi/4))
	d = avgDataTog(makeData(n,np.pi/2,-np.pi/4))
	CHSH = a+b+c-d
	return(CHSH)

def CHSHsep(n):
	a = avgDataSep(makeData(n,0,np.pi/4))
	b = avgDataSep(makeData(n,0,-np.pi/4))
	c = avgDataSep(makeData(n,np.pi/2,np.pi/4))
	d = avgDataSep(makeData(n,np.pi/2,-np.pi/4))
	CHSH = a+b+c-d
	return(CHSH)

def CHSHclassical(n):
	a = avgDataTog(makeDataC(n))
	b = avgDataTog(makeDataC(n))
	c = avgDataTog(makeDataC(n))
	d = avgDataTog(makeDataC(n))
	CHSH = a+b+c-d
	return(CHSH)
@nb.jit
def plotCHSHData(n):
	l = np.zeros(32)
	for i in range(0,32):
		phi = i*np.pi/16
		a = avgDataTog(makeData(n,0,np.pi/4+phi))
		b = avgDataTog(makeData(n,0,-np.pi/4+phi))
		c = avgDataTog(makeData(n,np.pi/2,np.pi/4+phi))
		d = avgDataTog(makeData(n,np.pi/2,-np.pi/4+phi))
		l[i] = a+b+c-d
	plt.plot(np.linspace(0,2*np.pi,32),l)
	plt.show()

def plotCHSHCalcd(n):
	l = np.linspace(0,2*np.pi,100)
	l1 = np.cos(-np.pi/4. - l) + 2*np.cos(np.pi/4.-l) - np.cos(3*np.pi/4 - l)
	plt.plot(l,l1)
	plt.show()

def main():
	import sys
	n = int(sys.argv[1])
	print("2sqrt(2) = ",2*np.sqrt(2))
	print("The CHSH correlator based on generated data is: ",CHSH(n))
	print("The CHSH correlator for each individual qubit is: ",CHSHsep(n))
	print("The classical correlator is: ",CHSHclassical(n))

if __name__ == "__main__":
	main()
