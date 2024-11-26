import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
[[ 12 157   4   4   1   1   0   0   0   5   0   0   0   3   7]
 [  0 201   1   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0  46 108   0  16   0   0   0   0   0   0   0   0   1   8]
 [  0   4   8 169   0   0   0   6   0   4   0   0   0  14   0]
 [  0  82   3   0 115   0   1   0   0   0   0   0   0   0   0]
 [  2  24   0   0   0 150  11   0   0   0   0   0   0   0   5]
 [  0  70   0   0   2   3 115   1   0   2   0   0   0   1   3]
 [  0 112   0   0   0   0   0 100   0   0   0   1   0   0   0]
 [  0   0   0   0   0   0   0   0 192   0   0   0   0   0   0]
 [  0   9   0   0   0   0   0   3   0 180   0   0   0   0   9]
 [  0 213   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0  95   0   0   0   0   0   0   0   0   0 100   0   0   0]
 [  0  14   0   0   0   0   0  10   5   0   0   0   0   2 154]
 [  0  89   0   0   0   0   0   0   0   0   0   0   0  92  41]
 [  0  10   0   0   0   0   0   0   0   0   0   0   0   2 197]]
"""
"""
map: {"0": "BENIGN", "1": "Bot", "2": "DDoS", "3": "DoS GoldenEye", "4": "DoS Hulk", "5": "DoS Slowhttptest", "6": "DoS slowloris", "7": "FTP-Patator", "8": "Heartbleed", "9": "Infiltration", "10": "PortScan", "11": "SSH-Patator", "12": "Web Attack \ufffd Brute Force", "13": "Web Attack \ufffd Sql Injection", "14": "Web Attack \ufffd XSS"}
"""
index = ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator',
         'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack & Brute Force',
         'Web Attack & Sql Injection', 'Web Attack & XSS']

import numpy as np

from util.plot.plot_confusion_matrix import plot_confusion_matrix

Data = [
    [12, 157, 4, 4, 1, 1, 0, 0, 0, 5, 0, 0, 0, 3, 7],
    [0, 201, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 46, 108, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 1, 8],
    [0, 4, 8, 169, 0, 0, 0, 6, 0, 4, 0, 0, 0, 14, 0],
    [0, 82, 3, 0, 115, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 24, 0, 0, 0, 150, 11, 0, 0, 0, 0, 0, 0, 0, 5],
    [0, 70, 0, 0, 2, 3, 115, 1, 0, 2, 0, 0, 0, 1, 3],
    [0, 112, 0, 0, 0, 0, 0, 100, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 3, 0, 180, 0, 0, 0, 0, 9],
    [0, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0],
    [0, 14, 0, 0, 0, 0, 0, 10, 5, 0, 0, 0, 0, 2, 154],
    [0, 89, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 41],
    [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 197]
]

classes_x = ['B', 'Bot', 'DDos', 'GE', 'DH', 'DS1', 'DS2', 'FTP', 'H', 'I', 'P', 'SSH', 'WB', 'WS', 'WX']
plot_confusion_matrix(np.array(Data), classes=index, classes_x=classes_x, normalize=True, title='')
