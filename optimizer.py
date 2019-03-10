# -*- coding: utf-8 -*-


import GWO as gwo
import csv
import numpy
import time


def selector(func_details,popSize,Iter):
   # function_name=func_details[0]
    lb=func_details[0]
    ub=func_details[1]
    dim=func_details[2]

    x=gwo.GWO(lb,ub,dim,popSize,Iter)
    return x


def param(a):
    param = {0: [0, 1, 24]}
    return param.get(a,"noting")


# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=1

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 7
Iterations= 7

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="./result/experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

for k in range(0, NumOfRuns):
    func_details = param(0)
    x = selector(func_details, PopulationSize, Iterations)
    if (Export == True):
        with open(ExportToFile, 'a', newline='\n') as out:
            writer = csv.writer(out, delimiter=',')
            if (Flag == False):  # just one time to write the header of the CSV file
                header = numpy.concatenate(
                    [["Optimizer", "objfname", "startTime", "EndTime", "ExecutionTime"], CnvgHeader])
                writer.writerow(header)
            a = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime], x.convergence])
            writer.writerow(a)
        out.close()
    Flag = True  # at least one experiment
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
