#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 00:45:49 2023

@author: joshuauhlfelder
"""

import csv

data_dir = '../HAM10000_images'
metadata = 'HAM10000_metadata.csv'
count = 0
with open(metadata, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        if row[2] == 'nv':
            count += 1
print(count)