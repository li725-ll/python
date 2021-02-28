import numpy
import scipy.special
import cv2
import os
import math
import xlwt
import matplotlib.pyplot as plt

class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  # 初始化输入节点、隐藏结点、输出结点、学习率
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
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
		return final_outputs

		pass


	def query(self, inputs_list):
		# 将输入转换为二维数组
		inputs = numpy.array(inputs_list, ndmin=2).T
		# 隐层输入信号的计算
		hidden_inputs = numpy.dot(self.wih, inputs)  # 计算的两个数组的点积 矩阵乘法
		# 计算隐藏层输出
		hidden_outputs = self.activation_function(hidden_inputs)  # 将数据带入激活函数得出输出
		final_inputs = numpy.dot(self.who, hidden_outputs)  # 计算的两个数组的点积 矩阵乘法
		final_outputs = self.activation_function(final_inputs)  # 将数据带入激活函数得出输出
		return final_outputs
		pass

def loss(rate, data):
	input_nodes = 10000  # 输入结点个数
	hidden_nodes = 110  # 隐藏层结点个数
	output_nodes = 3  # 输出结点个数
	learning_rate = rate  # 学习率
	n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)  # 创建神经网络对象
	f = xlwt.Workbook()
	sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
	i = 0
	for filename in os.listdir("training"):
		img = cv2.imread("training/" + filename, 0)  # 读图片
		img = numpy.uint8(numpy.clip((2 * img), 0, 255))
		resizeimg = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
		# print(resizeimg)
		inputs = (numpy.asfarray(resizeimg.flatten()) / 255.0 * 0.99) + 0.01
		targets = numpy.zeros(output_nodes) + 0.01
		if str(filename[3]) == 's':
			targets[0] = 0.99
		elif str(filename[3]) == 'j':
			targets[1] = 0.99
		elif str(filename[3]) == 'b':
			targets[2] = 0.99
		result = n.train(inputs, targets)
		y = -(targets[0]*math.log(result[0])+targets[1]*math.log(result[1])+targets[2]*math.log(result[2]))
		data.append(y)
		sheet1.write(i, 0, y)
		i += 1
		pass
	#f.save("datas/loss_function"+str(rate)+".xls")


rate = 0.6
for i in range(7):
	data = []
	loss(rate, data)
	plt.title = str(rate)
	plt.plot([x for x in range(0, 600)], data)
	plt.savefig("images/" + str(rate) + ".png")
	plt.show()
	rate /= 2
