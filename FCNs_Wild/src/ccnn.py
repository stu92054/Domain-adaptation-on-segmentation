# --------------------------------------------------------
# CCNN 
# Copyright (c) 2015 [See LICENSE file for details]
# Written by Deepak Pathak, Philipp Krahenbuhl
# --------------------------------------------------------

def __setup_path():
	import os, sys, inspect, numpy as np
	paths = ['.','..','../build/','../build/release','../build/debug']
	current_path = os.path.split(inspect.getfile( inspect.currentframe() ))[0]
	paths = [os.path.realpath(os.path.abspath(os.path.join(current_path,x))) for x in paths]
	paths = list( filter( lambda x: os.path.exists(x+'/lib/python/ccnn.so'), paths ) )
	ptime = [os.path.getmtime(x+'/lib/python/ccnn.so') for x in paths]
	if len( ptime ):
		path = paths[ np.argmax( ptime ) ]
		sys.path.insert(0, path+'/lib')
__setup_path()
del __setup_path
from python.ccnn import *
