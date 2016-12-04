import os
import matplotlib.pyplot as plt

train_acc = dict()
test_acc = dict()

directories = [x for x in sorted(os.listdir('.')) if (not os.path.isfile(x) and x != '.DS_Store')]

for directory in directories:
  infile = open(directory + "/output.txt", 'r')

  train_acc[directory] = []
  test_acc[directory] = []

  for line in infile.readlines():
      if line.find("acc") == -1: continue
      train_acc[directory].append(float(line.split()[6]))
      test_acc[directory].append(float(line.split()[12]))

  infile.close()

plt.clf()
for directory in directories:
  plt.plot(train_acc[directory], label=directory)
  plt.legend(loc="best")
  plt.title('Training Accuracy For Neural Networks')
  plt.ylabel('Accuracy (%)')
  plt.xlabel('Epoch')
  plt.savefig('train_acc.png')

plt.clf()
for directory in directories:
  plt.plot(test_acc[directory], label=directory)
  plt.legend(loc="best")
  plt.title('Test Accuracy For Neural Networks')
  plt.ylabel('Accuracy (%)')
  plt.xlabel('Epoch')
  plt.savefig('test_acc.png')
