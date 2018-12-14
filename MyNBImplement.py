import csv

## USING PYTHON 3.6 ##
## MUST HAVE VALID ARFF FILE ##
## THIS TOOL USE NAIVE BAYES WITH LAPLACIAN SMOOTHING ##

## Refer to:
## https://github.com/haloboy777
## https://machinelearningcoban.com/2017/08/08/nbc/?fbclid=IwAR0fn5txZ82qGPrWpFVP6g5m39ZVDVw3BqxhhYe8Uwx61Fal8crZjDO_ZKA


def toCsv(filename):
	f = open(filename, 'rt')
	content = f.readlines()
	data = False
	header = ""
	newContent = []
	for line in content:
		if not data:
			if "@attribute" in line.lower():
				attri = line.lower().split()
				columnName = attri[attri.index("@attribute")+1]
				header = header + columnName + ","
			elif "@data" in line.lower():
				data = True
				# Remove coma
				header = header[:-1]
				header += '\n'
				newContent.append(header)
		else:
			if "'" in line:
				line = line.replace("'","")
			newContent.append(line)
	return newContent

def readCsv(content):
	rows = csv.reader(content)
	headers = next(rows, None)
	column = {}
	for h in headers:
		column[h] = []
	outrow = []
	index = 0
	for row in rows:
		outrow.append({})
		for h, value in zip(headers, row):
			column[h].append(value)
			outrow[index][h] = value
		index+=1
	return (headers, column, outrow)

class Classifier:
	def __init__(self, train_name, test_name):
		self.content = toCsv(train_name)
		(self.header, self.column, self.row) = readCsv(self.content)
		self.classes = set(self.column[self.header[-1]])

		#Read test data
		(_, _, self.test_row) = readCsv(toCsv(test_name))

		#Probs
		self.prob = self.classesProbability()
		
		
	def classesProbability(self):
		classColumn = self.column[self.header[-1]]
		length = len(classColumn)
		self.prob = {x : classColumn.count(x)/length for x in set(classColumn)}
		return self.prob

	def classifyTest(self):
		# Run through every instance in test
		for row in self.test_row:
			row[self.header[-1]] = self.classify(row)
			print('Test ' + str(self.test_row.index(row) + 1) + ': ' + str(row[self.header[-1]]), end = '\n\n')
		
	def classify(self, instance):
		#TODO prob[c] = prob of instance to be in class c
		#Calculate prob of each attribute
		prob = {}
		for c in self.classes:
			prob[c] = self.probInstanceInClass(instance, c)
		#Get the maximum prob => class
		clas = max(prob, key=prob.get)
		return clas
	
	# prob = PI p(xi | c) * p(c)
	def probInstanceInClass(self, instance, clas):
		ans = self.prob[clas]
		
		classCount = self.column[self.header[-1]].count(clas)
		#Remove the class and the animal name
		attrs = self.header[1:-1]
		for attr in attrs:
			# Count train instance that has same attr with given instance and in the specified class
			# Use the fact that int(True) == 1
			attrCount = sum((r[attr] == instance[attr] and r[self.header[-1]] == clas) for r in self.row)
			# Multiply ans with the probability 
			# Smoothing by count individual values of that attribute
			valueCount = len(set(self.column[attr]))
			ans *= float(attrCount + 1)/(classCount + valueCount)
		return ans

	def printStatistic(self):
		print('Number of instance: ' + str(len(self.row)), end = '\n\n')
		print(str(len(self.header)) + ' Attributes: ' + str(self.header), end = '\n\n')
		print(str(len(self.classes)) + ' Classes: ' + str(self.classes), end = '\n\n')

		#Classes Probability	
		print('Classes probability: ' + str(self.prob), end = '\n\n')
		return
		

def main():	
	# Init and Statistics
	c = Classifier ('zoo.arff','zootest.arff')
	c.printStatistic()
	c.classifyTest()


if __name__ == "__main__":
	main()