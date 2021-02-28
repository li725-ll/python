#!/usr/bin/env python
# coding: utf-8

import numpy
import scipy.special
import matplotlib.pyplot
import cv2
import os

array_of_img = []

class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  # 初始化输入节点、隐藏结点、输出结点、学习率
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		# self.wih = (numpy.random.rand(self.hnodes, self.inodes)- 0.5)
		# self.who = (numpy.random.rand(self.onodes, self.hnodes)- 0.5)
		# 初始化时权重.0数组矩阵
		self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))  # 建立输入层和隐藏层的权重数组
		self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))  # 建立输出层和隐藏层的权重数组
		self.activation_function = lambda x: scipy.special.expit(x)  # 定义激活函数
		pass

	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		output_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T, output_errors)
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
										numpy.transpose(hidden_outputs))
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
										numpy.transpose(inputs))
		pass

	def query(self, inputs_list):
		# 将输入转换为二维数组
		inputs: object = numpy.array(inputs_list, ndmin = 2).T
		# 隐层输入信号的计算
		hidden_inputs = numpy.dot(self.wih, inputs)  # 计算的两个数组的点积 矩阵乘法
		# 计算隐藏层输出
		hidden_outputs = self.activation_function(hidden_inputs)  # 将数据带入激活函数得出输出
		final_inputs = numpy.dot(self.who, hidden_outputs)  # 计算的两个数组的点积 矩阵乘法
		final_outputs = self.activation_function(final_inputs)  # 将数据带入激活函数得出输出
		return final_outputs
		pass


input_nodes = 10000  # 输入结点个数
hidden_nodes = 110  # 隐藏层结点个数
output_nodes = 3  # 输出结点个数
learning_rate = 0.1  # 学习率
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)  # 创建神经网络对象
# print(numpy.asfarray(all_values[1:]))
for filename in os.listdir("training"):
	img = cv2.imread("training/" + filename, 0)  # 读图片
	img = numpy.uint8(numpy.clip((2 * img), 0, 255))
	resizeimg = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
	inputs = (numpy.asfarray(resizeimg.flatten()) / 255.0 * 0.99) + 0.01
	""" print(numpy.asfarray(resizeimg))
		matplotlib.pyplot.imshow(resizeimg, cmap='Greys', interpolation='None')
		matplotlib.pyplot.show()"""
	targets = numpy.zeros(output_nodes) + 0.01
	if str(filename[3]) == 's':
		targets[0] = 0.99
	elif str(filename[3]) == 'j':
		targets[1] = 0.99
	elif str(filename[3]) == 'b':
		targets[2] = 0.99
	n.train(inputs, targets)
	# print(targets)
	pass
print("训练完成！！")
# Testing
scorecard = []
for filename in os.listdir("training_son"):
	img = cv2.imread("training_son/" + filename, 0)  # 读图片
	correct_label = str(filename[3])
	img = numpy.uint8(numpy.clip((2 * img), 0, 255))
	resizeimg = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
	#image_array = numpy.asfarray(resizeimg)
	#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
	#matplotlib.pyplot.show()
	a = n.query((numpy.asfarray(resizeimg.flatten()) / 255.0 * 0.99) + 0.01)# 将0-255这个较大范围缩小到0.01-1.0
	num = numpy.argmax(a)
	if num== 0:
		lable = 's'
	elif num == 1:
		lable = 'j'
	elif num == 2:
		lable = 'b'
		pass
	if lable == correct_label:
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass

scorecard_array = numpy.asarray(scorecard)
print("训练子集的正确率= ", scorecard_array.sum() / scorecard_array.size)

for filename in os.listdir("text"):
	img = cv2.imread("text/" + filename, 0)  # 读图片
	correct_label = str(filename[3])
	img = numpy.uint8(numpy.clip((2 * img), 0, 255))
	resizeimg = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
	#image_array = numpy.asfarray(resizeimg)
	#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
	#matplotlib.pyplot.show()
	a = n.query((numpy.asfarray(resizeimg.flatten()) / 255.0 * 0.99) + 0.01)  # 将0-255这个较大范围缩小到0.01-1.0
	num = numpy.argmax(a)
	if num== 0:
		lable = 's'
	elif num == 1:
		lable = 'j'
	elif num == 2:
		lable = 'b'
		pass
	if lable == correct_label:
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass

scorecard_array = numpy.asarray(scorecard)
print("测试集的正确率 = ", scorecard_array.sum() / scorecard_array.size)  # share of correct answers

for filename in os.listdir("text_diferent"):
	img = cv2.imread("text_diferent/" + filename, 0)  # 读图片
	correct_label = str(filename[3])
	img = numpy.uint8(numpy.clip((2 * img), 0, 255))
	resizeimg = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
	#image_array = numpy.asfarray(resizeimg)
	#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
	#matplotlib.pyplot.show()
	a = n.query((numpy.asfarray(resizeimg.flatten()) / 255.0 * 0.99) + 0.01)  # 将0-255这个较大范围缩小到0.01-1.0
	num = numpy.argmax(a)
	if num == 0:
		lable = 'c'
	elif num == 1:
		lable = 'j'
	elif num == 2:
		lable = 'b'
		pass
	if lable == correct_label:
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass

scorecard_array = numpy.asarray(scorecard)
print("不同时间拍摄的图片准确率 = ", scorecard_array.sum() / scorecard_array.size)