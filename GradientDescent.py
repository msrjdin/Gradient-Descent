import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
#%matplotlib inline
#creating the function and plotting it 



class GD:
	def __init__(self, feature_count, y_obs, features, learningRate, maxSteps=1000, minStepSize=0.001, intercept=True):
		
		self.feature_count=feature_count
		self.y_obs=y_obs.values
		self.intercept=intercept
		self.coeff=np.linspace(1,1,feature_count)
		if intercept:
			features['intercept']=1
			self.coeff=np.append(self.coeff,0)
		self.features=features.values
		self.learningRate=learningRate
		
		
		self.maxSteps=maxSteps
		self.minStepSize=minStepSize

		
	def costFunction(self):
		cost=0
		for j in range(len(self.y_obs)):
			(y, f) = (self.y_obs[j], self.features[j])
			ff=np.array(f)
			#print(ff,y, sum(self.coeff*ff))
			#Indidividual Squared Residual (y-(ax1+bx2+cx3.. +intercept))
			c=y-sum(self.coeff*ff)
			#print(c)
			cost+=c**2
		return cost
	
	def gradient(self):
		slope=[]
		for i in range(self.feature_count):
			c=0
			for j in range(len(self.y_obs)):
				#print(j)
				(y,f) = (self.y_obs[j], self.features[j])
				ff=np.array(f)
				#print((-2)*self.coeff[i]*(y-sum(self.coeff*ff)))
				c+=(-2)*self.coeff[i]*(y-sum(self.coeff*ff))
			if self.intercept:
				#print((-1)*(y-sum(self.coeff*ff)))
				c+=(-1)*(y-sum(self.coeff*ff))
			slope.append(c)
		return np.array(slope)
	
	def descent(self):
		stepSize=np.linspace(1,1,self.feature_count)
		if self.intercept:
			stepSize=np.append(stepSize,1)
		s=0
		costPoints=[]
		print(self.coeff)
		CFS=[]
		
		while True: 
			print(s)
			#print(CFS)
			costPoints.append(self.costFunction())
			CFS.append(self.coeff.copy())
			#print(costPoints)
			t=self.gradient()
			stepSize=t*self.learningRate
			#print(t)
			self.coeff-=stepSize
			#print(self.coeff)
			s=s+1
			print(stepSize)
			if s>self.maxSteps or (abs(stepSize)<self.minStepSize).any():
				break
		
		return [self.coeff,costPoints, CFS]
	
df=pd.read_csv('data.txt')
g=GD(1, df['y'], df[['f1']], .1, 1000, 0.0001,intercept=True)
coefficients,costPoints, CFS=g.descent()
print(coefficients)
plt.plot(pd.DataFrame(CFS)[0], costPoints)
				
					
		
plt.show()
	

	