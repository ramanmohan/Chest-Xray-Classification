#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:40:59 2019

@author: ramanmohan
"""
import glob
import os

f = open("augmented.txt", 'w')

path1 = './images/healthy_augmented'

for file in glob.glob(os.path.join(path1, '*.png')):
    f.write("healthy %s" %file)
    f.write('\n')
    
path2 = './images/unhealthy_augmented'

for file in glob.glob(os.path.join(path2, '*.png')):
    f.write("unhealthy %s" %file)
    f.write('\n')
    