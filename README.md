# Stochastic Simulation Assignment 02: Discrete Event Simulation (Simple Queues)

### License:
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains Python scripts developed for a the second assignment of Stochastic Simulation at the University of Amsterdam, 2023. The project focuses on the simulating various queuing types with different number of servers and system loads. It utilises the [SimPy framework](https://simpy.readthedocs.io/en/latest/).

## File Descriptions

### `simulation.py`
This script contains the main functions for the simulation. It enables to run five different types of queue (M/M/m (FIFO), M/d/m, hyperexponential, M/M/m (Shortest Job First) and M/M/m/K) for different service loads and server numbers. The output is in a .csv file containing all metrics and statistics.

### `plots.py`
It takes the .csv output from `simulation.py` and creates a variety of plots, for instance the mean waiting time of the queues for different queuing types, service numbers and loads as heatmaps and comparison line plots as well as the distributions of individual stochastic simulations. It can also perform a comparison between all queing models for a set of parameters.

## Usage
These scripts were run with Python 3.11.0 on MacOS Ventura. 

### Requirements:
* matplotlib==3.7.1
* numpy==1.24.3
* pandas==1.5.3
* scipy==1.10.1
* seaborn==0.13.0
* simpy==4.1.1

## Contact

joana.costaesilva@student.uva.nl 
balint.szarvas@student.uva.nl
sandor.battaglini-fischer@student.uva.nl

---

Developed by Joana Costa e Silva, Bálint Szarvas and Sándor Battaglini-Fischer.

