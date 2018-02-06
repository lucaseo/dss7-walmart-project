import scipy as sp
print("success import scipy")
import pandas as pd
print("success import pandas")
import numpy as np
print("success import numpy")
import sympy
sympy.init_printing(use_latex='mathjax')
print("success import sympy")
import platform
import matplotlib.pyplot as plt
print("success import matplotlib.pyplot")
from numdifftools import Jacobian, Hessian

path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
	print("Hangul OK in your MAC !!!")
	rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
	font_name = font_manager.FontProperties(fname=path).get_name()
	print("Hangul OK in your Windows !!!")
	rc('font', family = font_name)
else:
	print("Unknown system... sorry~~~")

plt.rcParams['axes.unicode_minus'] = False
