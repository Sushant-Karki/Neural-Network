import numpy as nm
import random
import csv
import math
import time
import operator
import copy
import matplotlib.pyplot as plt
import sys

NInput = 784 #int(sys.argv[1])
NHidden = 30 #int(sys.argv[2])
NOutput = 10 #int(sys.argv[3])
TrainDigitX = sys.argv[4]
TrainDigitY = sys.argv[5]
TestDigitX = sys.argv[6]
PredictDigitY = sys.argv[7]

# modify the program later to include inputs from command line.
# Wight1 is for weight used in links from input layer to hidden layer
Weight1 = []
Weight2 = []
Weight1 = [[float(random.normalvariate(0,1)) for column in range(NHidden)] for row in range(NInput)]
Weight2 = [[float(random.normalvariate(0,1)) for column in range(NOutput)] for row in range(NHidden)]

#python neural_network.py NInput NHidden NOutput TrainDigitX.csv.gz
#TrainDigitY.csv.gz TestDigitX.csv.gz PredictDigitY.csv.gz

# you may use the code below to help with the testing of the algoritm because mine is too slow.
# the code below reads the weight stored

#for row in open("weight1.txt","r"):
#    row = row.split(", ")
#    row[0] = row[0].replace("[","")
#    row[-1] = row[-1].replace("]\n","")
#    row = [float(i) for i in row]
#    Weight1.append(row)

#for row in open("weight2.txt","r"):
#    row = row.split(", ")
#    row[0] = row[0].replace("[","")
#    row[-1] = row[-1].replace("]\n","")
#    row = [float(i) for i in row]
#    Weight2.append(row)

bias1 = float(random.normalvariate(0,1))
bias2 = float(random.normalvariate(0,1))


n_samples = 50000
n_epochs = 30
batch_size = 5
n_batches = round(n_samples/batch_size)
hidden_layer = []
output_layer = []
learning_rate = 3
pyplot = [[0] for j in range(2)]


def get_input(file):
    image = file.readline().split(",")
    image = [float(convert_into_int) for convert_into_int in image]
    return image


def Hidden_layer_operation(hidden_layer,image,Weight1,bias1):
    hidden_layer = nm.dot(image,Weight1)
    for i in range(len(hidden_layer)):
        hidden_layer[i] += bias1
    for value in range(len(hidden_layer)):
        hidden_layer[value] = 1/(1+ math.exp(-hidden_layer[value]))
    return hidden_layer


def Output_layer_operation(output_layer,hidden_layer,Weight2,bias2):
    output_layer = nm.dot(hidden_layer,Weight2)
    for i in range(NOutput):
        output_layer[i] += bias2
    for value in range(len(output_layer)):
        output_layer[value] = 1/(1+ math.exp(-output_layer[value]))
    return output_layer


def predicting(hidden_layer, Weight1, Weight2, bias1, bias2):

    test_file = open("TestDigitX2.csv","r")
    write_lable = open("PredictDigitY2.csv","w",newline='')
    writer = csv.writer(write_lable)

    for test_sample in range(5000):

        # Getting input
        image = get_input(test_file)

        # Finding inputs from hidden layer
        hidden_layer = []
        hidden_layer = Hidden_layer_operation(hidden_layer,image,Weight1,bias1)

        # Finding outputs from output layer
        output_layer = []
        output_layer = Output_layer_operation(output_layer,hidden_layer,Weight2,bias2)

        biggest = 0
        for i in range(NOutput):
            if output_layer[i] > biggest:
               biggest = output_layer[i]
               prediction = copy.deepcopy(i)

        writer.writerow(str(prediction))

    write_lable.close()


def testing(hidden_layer, Weight1, Weight2, bias1, bias2, i_epoch):
    global pyplot
    global TestDigitX
    global TestDigitY

    test_file = open(TestDigitX,"r")
    test_lable = open("TestDigitY.csv","r")
    write_lable = open(PredictDigitY,"w",newline='')
    writer = csv.writer(write_lable)
    cost = 0
    prediction = 0
    Accuracy = 0
    Accuracy_percentage = 0

    for test_sample in range(5000):

        # Getting input
        image = get_input(test_file)

        # Finding lable
        read = int(test_lable.readline())
        lable = [i*0 if i!=read else 1 for i in range(NOutput)]

        # Finding inputs from hidden layer
        hidden_layer = []
        hidden_layer = Hidden_layer_operation(hidden_layer,image,Weight1,bias1)

        # Finding outputs from output layer
        output_layer = []
        output_layer = Output_layer_operation(output_layer,hidden_layer,Weight2,bias2)

        biggest = 0
        for i in range(NOutput):
            if output_layer[i] > biggest:
               biggest = output_layer[i]
               prediction = copy.deepcopy(i)


        writer.writerow(str(prediction))

        if read == prediction:
            Accuracy += 1

        # differnce
        difference = []
        difference = copy.deepcopy(output_layer)
        difference[read] = difference[read]-1

        # Cost function
        cost += sum(difference)**2

    cost = cost/(2*5000)
    print("...........................................")
    print("cost of test data  ", cost)

    Accuracy_perecntage = (Accuracy/5000)*100
    print("Accuracy is : ", Accuracy_perecntage, "%")

    pyplot[0].append(i_epoch)
    pyplot[1].append(cost)
    plt.title("epoch vs cost")
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.plot(pyplot)
    plt.show()

    test_file.close()
    test_lable.close()
    write_lable.close()



def Neural_Network(n_epochs,batch_size,n_batches, Weight1, Weight2, bias1, bias2, learning_rate):
    global pyplot
    global TrainDigitX
    global TrainDigitY

    for i_epoch in range(n_epochs):
        highest_weight = 0
        for_lable = 0
        print("epoch no. ", i_epoch)

        file = open(TrainDigitX,"r")
        lable_file = open(TrainDigitY,"r")

        for i_batch in range(n_batches):

            sum_gradW1 = [[0 for column in range(NHidden)] for row in range(NInput)]
            sum_gradW2 = [[0 for column in range(NOutput)] for row in range(NHidden)]

            for current_batch in range(batch_size):

                # Getting the input values
                image = get_input(file)


                # Finding lable
                read = int(lable_file.readline())
                lable = [i*0 if i!=read else 1 for i in range(NOutput)]


                # Finding inputs from hidden layer
                hidden_layer = []
                hidden_layer = Hidden_layer_operation(hidden_layer,image,Weight1,bias1)


                # Finding outputs from output layer
                output_layer = []
                output_layer = Output_layer_operation(output_layer,hidden_layer,Weight2,bias2)



                # Calculating error
                total_error = 0
                for i in range(NOutput):
                    total_error += .5*(lable[i]-output_layer[i])**2


                # The backward pass
                # for Weight2
                for i in range(NOutput):
                    for j in range(NHidden):
                        gradW_x = (output_layer[i]-lable[i])*(output_layer[i]*(1-output_layer[i]))*(hidden_layer[j])
                        sum_gradW2[j][i] += gradW_x


                # for Weight1
                for i in range(NHidden):
                    grad_error_by_output = 0
                    for k in range(NOutput):
                        grad_error_by_output += (output_layer[k]-lable[k])*(output_layer[k]*(1-output_layer[k]))*(Weight2[i][k])
                    for j in range(NInput):
                        gradW_x = grad_error_by_output*(hidden_layer[i]*(1-hidden_layer[i]))*(image[j])
                        sum_gradW1[j][i] += gradW_x


            # Updating the Weights
            for i in range(NInput):
                for j in range(NHidden):
                    Weight1[i][j] = Weight1[i][j] - learning_rate*(sum_gradW1[i][j]/batch_size)


            for i in range(NHidden):
                for j in range(NOutput):
                    Weight2[i][j] = Weight2[i][j] - learning_rate*(sum_gradW2[i][j]/batch_size)


            for i in range(NOutput):
                if lable[i] == 1:
                    
                    if output_layer[i] > highest_weight:
                        highest_weight = copy.deepcopy(output_layer[i])
                        for_lable = copy.deepcopy(i)

            # Storing the weights for future use
            write_weight_1 = open("weight1.txt","w")
            for i in Weight1:
                write_weight_1.write(str(i))
                write_weight_1.write('\n')
            write_weight_2 = open("weight2.txt","w")
            for i in Weight2:
                write_weight_2.write(str(i))
                write_weight_2.write('\n')
            write_weight_1.close()
            write_weight_2.close()

            print()


        file.close()
        lable_file.close()

        testing(hidden_layer, Weight1, Weight2, bias1, bias2, i_epoch)



Neural_Network(n_epochs,batch_size,n_batches, Weight1, Weight2, bias1, bias2, learning_rate)
predicting(hidden_layer, Weight1, Weight2, bias1, bias2)
