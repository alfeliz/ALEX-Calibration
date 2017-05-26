#!/usr/bin/env python
#Calibration procedure for ALEX circuit
#coding: latin-1
#Version 1.00

import os #File management
import numpy as np #Numerical work in Python. Yeah!!!
import matplotlib.pyplot as plt
# (http://stackoverflow.com/questions/710551/import-module-or-from-module-import)
import csv #CSV files reading and writing module.
import scipy.integrate as inte #Numerical integration. YOU NEED TO INSTALL THE SCIPY PACKAGE.
from scipy.optimize import leastsq
############################################################################################################################
#To adapt this code within ALEX, take care of the channels transformations....
############################################################################################################################



###
# Definition of important variables
###

ms = 1e-3
us = 1e-6

V2Res = []
V3Res = []
volt = []
time = []
current = []
curr_not_alig = []
der_curr =[]



###
# Opening channels data and transforming them from the CSV files
###

with open("ALL_CH1.csv") as CH1csv:
	dialect = csv.Sniffer().sniff(CH1csv.read(1024))
	CH1csv.seek(0)
	reader = csv.reader(CH1csv, dialect)
	for row in reader:
		time.append(float(row[0])) #I put time in seconds here!!!
		der_curr.append(9.36e9*float(row[1])) #Current der. in A/s

with open("ALL_CH2.csv") as CHcsv:
	dialect = csv.Sniffer().sniff(CHcsv.read(1024))
	CHcsv.seek(0)
	reader = csv.reader(CHcsv, dialect)
	for row in reader:
		V2Res.append(1359*float(row[1])) #Volts in the ALEX

with open("ALL_CH3.csv") as CHcsv:
	dialect = csv.Sniffer().sniff(CHcsv.read(1024))
	CHcsv.seek(0)
	reader = csv.reader(CHcsv, dialect)
	for row in reader:
		V3Res.append(2400*float(row[1])) #Volts in the ALEX
for i in range(len(V2Res)):
	volt.append(V2Res[i]-V3Res[i])

#Shot name for the output file
shot_name = str(raw_input("Disparo? "))

#Current problem persists in the alignment. No clear reason for it.
curr_not_alig = inte.cumtrapz(der_curr, time, initial=0) #Not correctly aligned!!!

#Placing current rightly:
pol = np.polyfit(time,curr_not_alig,1)

current = curr_not_alig - np.polyval(pol,time) + pol[1]



###
#Finding ALEX support parameters by adjusting the voltage to the ideal circuit
###

#We make a stack of the currnet and its derivative vectors to have a matrix of data to adjust
x = np.vstack([current, der_curr]).T

#Using just Numpy to solve the problem, not sklearn:
#Linear adjust of Voltage to the current amd its derivative. 
#The parameters are R and L, after simple transformations
#np.linalg.lstsq  returns a list with:
# parameters_results[0], sum of residues[1], rank of matrix x[2], singular values of x[4]
#First with the data from 2Res divider, R and L are the addition of the support and earth values
sum_par = np.linalg.lstsq(x, np.array(V2Res))[0] #Support and earth values.

#Now with data from 3Res divider, R and L are the support values only.
x2 = np.vstack([current, der_curr]).T
par = np.linalg.lstsq(x2, np.array(volt), rcond=1e-2)[0] #Support values.



###
#Finding ALEX L and R, the other part of the circuit by
#defining the functions to fit C+B*exp(-alpha*(x-x0))*cos(omega*(x-x0)),
#which is the current derivative. Also possible to use this to adjust the Rogowsky.
###

#Initial guess for parameters:
#Meaning of the vector, check a few lines down its transformation: 
# guess_par[0] ---> Baseline for the current derivative.
# guess_par[1] ---> Mulpitplication factor for the exponential oscillatory decay
# guess_par[2] ---> Decay contant of exponential.
# guess_par[3] ---> Timing of current initial respect the scope zero time.
# guess_par[4] ---> Frequency of the oscillation.
#REMEMBER THAT EVERYTHING HERE HAS UNITS. IN HERE SI UNITS
guess_par_cir = [np.mean(der_curr[0:10]), abs(max(der_curr))*0.1, 1e-3/(abs(time[0]-time[1])), 1.67e-6, 1.74e6]

#Function of derivative current (direct channel signal):
func = lambda t,par_cir:par_cir[0] + par_cir[1]*np.exp(-par_cir[2]*(t-par_cir[3]))*np.cos(par_cir[4]*(t-par_cir[3]))

#Function to optimize (function minus der_curr[current derivative]):
optimize = lambda par_cir: func(time[400:],par_cir) - der_curr[400:]

#Optimization with leastsq function:
adj_par_cir = leastsq(optimize, guess_par_cir)[0]

#Calculation of derivative current values from the adjustment
adjusted =[]
for i in time[400:]:
	adjusted.append(func(i,adj_par_cir))

#Statistical Error(high because of the problem with the end part of the current)
err_vec =[]
for i in range(len(time[400:])):
	err_vec.append((der_curr[i]-adjusted[i])**2)

error = np.sqrt(sum(err_vec)/len(time[400:]))

#Making sense of the parameters related with R an dL of the circuit:
#Inductance of the circuit
Cir_Inductance = 1 / ( 2.2e-6 * ( adj_par_cir[4]**2 + adj_par_cir[3]**2) ) #Henri
#The value of ALPHA(adj_par[3]) is almost zero...
#Resistance of the circuit
Cir_Resistance = 2 * adj_par_cir[3] * Cir_Inductance #Ohmns



###
# Storing the obtained parameters in a text file
###

with open(shot_name+"-cali.txt","w") as save_file:
	save_file.write("Calibration values for "+shot_name+"(S.I. Units)\n\n")
	save_file.write("Support results:\n")
	save_file.write("R_sop\t\tL_sop:\n")
	save_file.write("{:0.3e}".format(par[0])+"\t\t"+"{:0.3e}".format(par[1])+"\n")
	save_file.write("Earth pole results:\n")
	save_file.write("R_tie\t\tL_tie:\n")
	save_file.write("{:0.3e}".format(sum_par[0]-par[0])+"\t\t"+"{:0.3e}".format(sum_par[1]-par[1])+"\n")
	save_file.write("\nCircuit Results:\n")
	save_file.write("R_cir\t\tL_cir:\n")
	save_file.write("{:0.3e}".format(Cir_Resistance)+"\t\t"+"{:0.3e}".format(Cir_Inductance)+"\n")



###
# Saving a graph with the adjustment
###

plt.plot(time,der_curr,"r",  time[400:], adjusted, "k")
plt.legend(["Current derivative", "Adjusted"])
plt.title("ALEX calibration with shot "+shot_name)
plt.ylabel("A/s")
plt.xlabel("Seconds")
plt.savefig(shot_name+"-cir-adj.pdf", bbox_inches='tight')



#And tha...tha...that`s all folks!
