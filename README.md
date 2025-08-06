# OD MATRIX CALIBRATION

## 1. Introduction
This repository contains optimization/approximation algorithms to solve the offline OD (Origin-Destination) Matrix Calibration problem. Simulation-based optimization algorithms such as SPSA (Simultaneous Perturbation Stocahstic Approximation, Spall 1998), FDSA (Finite Difference SA, Spall 1998) and PC-SPSA (Principal Components SPSA, Qurashi 2020) can be conducted for a toy network. In "data_tabular" directory, a true OD matrix and edge flows are stored. A biased OD matrix is also located in the same directory and it is reduced version of the true OD matrix. Our algorithms are aimed to persuit (i.e. approximate) the true OD matrix from the biased one by minimizing error between true link flow and simulated link flow. For the simulation, algoritmes use either the SUMO or pretrained the Graph Neural Network (GNN) model.

## 2. Prerequisites
All algorithms are purely written in Python. So, if you are using recent release of Anaconda, there won't be severe issues to execute them. But, you may need to install below packages for applications of the Graph Neural Network Model and reporting manoeuvres. Also, the SUMO is needed to be installed if you want to use transport simulator rather than pretrained GNN model.</br>
+ Pandas & Numpy for data post processing.
+ Plotly & Dash for visualization of the results and parameters.
+ Scikit-learn for statistical methods.
+ Pytorch for general application of the machine learning model.
+ Pytorch Geometric (PyG) for application of the GNN.
+ The SUMO (Simulation of Urban MObility) for the transport simulator : <a href= "https://sumo.dlr.de/docs/index.html"> LINK for SUMO </a>

## 3. Toy network
A toy network is a simple traffic network all algorithms are concerning. Is is assumed that 4 blocks of residential area surrounded with links consist of two edges for both directions. And, each edge consists of two lanes. Also, two external links are set up for external influx and out flux of traffic flow. With this network, OD counts (i.e. Nr of Vehicle) and traffic flows (i.e. Veh/hr) are assinged for each edge of links. IMPORTANT: All edges in the network are also nodes in the GNN as actual traffic assignment is conducted on each edge.</br>

<p align="center"><img src="https://github.com/hosig0204/OD_Matrix_Calibration/blob/7da13d627261392d2828485f8cdc9d48caf39b53/static/images/toyNetworkODMatrix.jpg" width="800"></p>

## 4. For those who are intersted in SPSA
If you wan to see how the SPSA is working on approximating the true OD matrix, you can directly go to a python file "spsa_operation.py" and excute it. All parameters and reporting configurations are already set for an example case using pretrained GNN model. However, you might play with all prameters as your demands such as using the SUMO rather than the GNN, storing historical OD-flow data and adjusting iteration numbers. Keep in mind that using the SUMO will require a lot of time to excute a whole process.

## 5. For those who are intersted in FDSA
As FDSA requires a lot of perturbation for gradient estimation, this algorithm only untilzes the GNN model instead of the SUMO. You can go to a notebook "fdsa.ipynb" and play a whole process. But, pre-defined parameters are not optimized and results are generally poor at this moment. 

## 6. For those who are intersted in PC-SPSA
This algorithm is in development and only uses the SUMO for objective evaluation steps. So, a lot of time required for execution. However, you might see how it works after tweaking some parameters (e.g. iteration numbers). Notably, the PCA (Principal Components Analysis) is applied to reduce dimension of the OD matrix during approximation. For your trial, please follow these steps.
+ Go to a notebook "biased_OD_gen.ipynb" and execute it so that historical biased OD matrices can be populated. Currently, a total 100 matrices will be created and stored.
+ After having historical biased OD matrices, you can go to a notebook "pca_spsa.ipynb" and excute it. It will build PCA transformation object before approximation begins.
