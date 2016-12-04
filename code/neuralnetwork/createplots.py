import os
import matplotlib.pyplot as plt

for directory in os.listdir('.'):
  if os.path.isfile(directory): continue

  infile = open(directory + "/output.txt", 'r')

  train_acc = []
  test_acc = []

  for line in infile.readlines():
      if line.find("acc") == -1: continue
      train_acc.append(float(line.split()[6]))
      test_acc.append(float(line.split()[12]))

  infile.close()

  plt.clf()
  plt.plot(train_acc)
  plt.plot(test_acc)
  plt.legend(["training accuracy", "testing accuracy"], loc="best")
  plt.savefig(directory + '/' + directory + '_plot.png')
