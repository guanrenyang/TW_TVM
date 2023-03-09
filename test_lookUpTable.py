from AutoSchedule.tw_auto_scheduler import  *
import numpy as np
import math, os, json, time

add_to_lookUpTable(10, 70, [0.3, 0.1])
add_to_lookUpTable(10, 80, [0.5, 0,3])
add_to_lookUpTable(20, 90, [0.9, 0.1])

print(read_from_lookUpTable(10, 70))
print(read_from_lookUpTable(10, 90))
print(read_from_lookUpTable(20, 90))s