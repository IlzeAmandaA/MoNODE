import numpy as np
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def read_amc(fname, crop=False):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    while content[0]!='1':
        content = content[1:]
    L = 30 # number of lines for each observed time point
    N = int(len(content)/L)
    D = 62
    data = np.zeros((N,D))
    for i in range(N):
        arr = [content[i*L+j].split()[1:] for j in range(1,L)]
        flat_arr = [float(item) for subarr in arr for item in subarr]
        data[i,:] = flat_arr
    if crop:
        idx_ = [a for subarr in [np.arange(0,31),np.arange(36,43),np.arange(48,54),np.arange(55,61)] for a in subarr]
        data = data[:,idx_]

    return data


def save_amc(D,fname='pred.amc'):
	if D.shape[1] != 62:
		raise ValueError('Input matrix does not have 62 dimensions.')
	with open(fname,'w') as f:
		f.write('#!Python matrix to amc conversion\n');
		f.write(':FULLY-SPECIFIED\n');
		f.write(':DEGREES\n');
		for frame in range(D.shape[0]):
			f.write('{:d}\n'.format(frame+1))
			f.write('root {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(*list(D[frame,0:6])))
			f.write('lowerback {:f} {:f} {:f}\n'.format(*list(D[frame,6:9])))
			f.write('upperback {:f} {:f} {:f}\n'.format(*list(D[frame,9:12])))
			f.write('thorax {:f} {:f} {:f}\n'.format(*list(D[frame,12:15])))
			f.write('lowerneck {:f} {:f} {:f}\n'.format(*list(D[frame,15:18])))
			f.write('upperneck {:f} {:f} {:f}\n'.format(*list(D[frame,18:21])))
			f.write('head {:f} {:f} {:f}\n'.format(*list(D[frame,20:23])))
			f.write('rclavicle {:f} {:f}\n'.format(*list(D[frame,24:26])))
			f.write('rhumerus {:f} {:f} {:f}\n'.format(*list(D[frame,26:29])))
			f.write('rradius {:f}\n'.format((D[frame,29])))
			f.write('rwrist {:f}\n'.format((D[frame,30])))
			f.write('rhand {:f} {:f}\n'.format(*list(D[frame,31:33])))
			f.write('rfingers {:f}\n'.format((D[frame,33])))
			f.write('rthumb {:f} {:f}\n'.format(*list(D[frame,34:36])))
			f.write('lclavicle {:f} {:f}\n'.format(*list(D[frame,36:38])))
			f.write('lhumerus {:f} {:f} {:f}\n'.format(*list(D[frame,38:41])))
			f.write('lradius {:f}\n'.format((D[frame,41])))
			f.write('lwrist {:f}\n'.format((D[frame,42])))
			f.write('lhand {:f} {:f}\n'.format(*list(D[frame,43:45])))
			f.write('lfingers {:f}\n'.format((D[frame,45])))
			f.write('lthumb {:f} {:f}\n'.format(*list(D[frame,46:48])))
			f.write('rfemur {:f} {:f} {:f}\n'.format(*list(D[frame,48:51])))
			f.write('rtibia {:f}\n'.format((D[frame,51])))
			f.write('rfoot {:f} {:f}\n'.format(*list(D[frame,52:54])))
			f.write('rtoes {:f}\n'.format((D[frame,54])))
			f.write('lfemur {:f} {:f} {:f}\n'.format(*list(D[frame,55:58])))
			f.write('ltibia {:f}\n'.format((D[frame,58])))
			f.write('lfoot {:f} {:f}\n'.format(*list(D[frame,59:61])))
			f.write('ltoes {:f}\n'.format((D[frame,61])))
                        