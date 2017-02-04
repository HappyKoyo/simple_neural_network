#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
import scipy.io

def Sigmoid(z):
	return 1 / (1 + np.exp(-1 * z))

def SigmoidGradient(z):#シグモイド関数微分
	return Sigmoid(z) * (1-Sigmoid(z))

def ToMatrix(vector_y):
	matrix_y = np.zeros((5000,10))
	for i in xrange(0,5000):
		matrix_y[i,vector_y[i]-1] = 1
	return matrix_y	#matrix(5000,10)

def InitTheta():#ランダム初期化
	e = 0.12#乱数範囲(-0.12~0.12)
	theta1 = np.random.rand(25,401)*e
	theta2 = np.random.rand(10,26)*e
	return theta1,theta2

def ForwardPropagete(x,y,m,theta1,theta2):
	x = np.concatenate((np.ones((m,1)),x),axis = 1)	#バイアス項追加 X(5000,401))
	z2 = np.dot(x,theta1.T)							#(5000,401)*(401,25) = z2(5000,25)
	a2 = Sigmoid(z2)
	a2 = np.concatenate((np.ones((m,1)),a2),axis = 1)#バイアス項追加 a2(5000,26)
	z3 = np.dot(a2,theta2.T)						#(5000,26)*(26,10) = z3(5000,10)
	h = Sigmoid(z3)									#仮説h(hypothesis)
	return h,a2,z2

def BackPropagate(x,y,m,theta1,theta2,a2,z2,h):
	delta3 = h - y									#δ3(5000,10)
	no_bias_theta2 = np.delete(theta2,0,1)			#バイアス項係数削除（10,25）
	g_dash = SigmoidGradient(z2)					#g'(5000,25)
	delta2 = np.dot(delta3,no_bias_theta2) * g_dash	#(5000,10)*(10,25).*(5000*25) = δ2(5000,25)
	x = np.concatenate((np.ones((m,1)),x),axis = 1)	#バイアス項追加 X(5000,401)
	large_delta1 = np.dot(x.T,delta2)/m				#(401,5000)*(5000,25) = Δ1(401,25) = パラメータの修正値
	large_delta2 = np.dot(a2.T,delta3)/m			#(26,5000)*(5000*10) = Δ2(26,10) = パラメータの修正値
	'''GradientChecking(プログラム診断用)
	x = np.delete(x,0,1)							#元のデータ長に戻す
	print GradientCheck(x,y,m,h,theta1,theta2,large_delta1,large_delta2)
	#'''
	learning_rate = 1								#学習率λ = 1
	theta1 = theta1 - learning_rate * large_delta1.T#Θ1(25,401)
	theta2 = theta2 - learning_rate * large_delta2.T#Θ2(10,26)
	return theta1,theta2

def CostFunction(y,m,h):#(5000*10)
	return np.sum(-1*y*np.log(h)-(1-1*y)*np.log(1-h))/m

def GradientCheck(x,y,m,h,theta1,theta2,large_delta1,large_delta2):
	e = 0.0001#epsilon
	e_app1 = np.zeros((25,401))#epsilon_approx　for theta1
	e_app1[0,0] = e
	e_app2 = np.zeros((10,26))#epsilon_approx　for theta2
	e_app2[0,0] = e
	h_eplus1 ,a2,z2 = ForwardPropagete(x,y,m,theta1+e_app1,theta2)
	h_eminus1,a2,z2 = ForwardPropagete(x,y,m,theta1-e_app1,theta2)
	h_eplus2 ,a2,z2 = ForwardPropagete(x,y,m,theta1,theta2+e_app2)
	h_eminus2,a2,z2 = ForwardPropagete(x,y,m,theta1,theta2-e_app2)
	g_app1 = (CostFunction(y,m,h_eplus1)-CostFunction(y,m,h_eminus1))/(2*e)#擬似微分
	g_app2 = (CostFunction(y,m,h_eplus2)-CostFunction(y,m,h_eminus2))/(2*e)#擬似微分
	return g_app1-large_delta1[0,0],g_app2-large_delta2[0,0]

def Test(x,y,m,theta1,theta2):
	score = 0
	h,a2,z2 = ForwardPropagete(x,y,m,theta1,theta2)
	for i in xrange(0,100):
		problem_num = int(np.random.rand()*4999)#問題番号
		if np.argmax(h[problem_num]) == np.argmax(y[problem_num]):
			score += 1
	print score,'point'

if __name__ == '__main__':
	mat = scipy.io.loadmat("ex4data1.mat")#バイナリサンプルデータ
	x = np.array(mat['X'])	#20*20画像データ*5000 matrix(5000,400)
	y = np.array(mat['y'])	#0~9数字データ*5000   vecter(5000,1)
	m = len(y)				#データ長
	y = ToMatrix(y)
	theta1,theta2 = InitTheta()
	for i in xrange(0,3000):#学習回数
		h,a2,z2 = ForwardPropagete(x,y,m,theta1,theta2)
		theta1,theta2 = BackPropagate(x,y,m,theta1,theta2,a2,z2,h)
		print i,CostFunction(y,m,h)
	Test(x,y,m,theta1,theta2)
