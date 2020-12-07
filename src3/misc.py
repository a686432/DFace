import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib

from IPython import embed

class Logger(object):
	def __init__(self):
		self._logger = None

	def init(self, logdir, name='log'):
		if self._logger is None:
			import logging
			if not os.path.exists(logdir):
				os.makedirs(logdir)
			log_file = os.path.join(logdir, name)
			if os.path.exists(log_file):
				os.remove(log_file)
			self._logger = logging.getLogger()
			self._logger.setLevel('INFO')
			fh = logging.FileHandler(log_file)
			ch = logging.StreamHandler()
			self._logger.addHandler(fh)
			self._logger.addHandler(ch)

	def info(self, str_info):
		self.init('./', 'tmp.log')
		self._logger.info(str_info)
logger = Logger()

#print = logger.info
def ensure_dir(path, erase=False):
	if os.path.exists(path) and erase:
		print("Removing old folder {}".format(path))
		shutil.rmtree(path)
	if not os.path.exists(path):
		print("Creating folder {}".format(path))
		os.makedirs(path)

def load_pickle(path):
	begin_st = time.time()
	with open(path, 'rb') as f: 
		print("Loading pickle object from {}".format(path))
		#v = pkl.load(f,  encoding='iso-8859-1')
		v = pkl.load(f)
	print("=> Done ({:.4f} s)".format(time.time() - begin_st))
	return v

def dump_pickle(obj, path):
	with open(path, 'wb') as f:
		print("Dumping pickle object to {}".format(path))
		pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
	import sys
	import os
	import subprocess
	import re
	import time
	import numpy as np
	if 'CUDA_VISIBLE_DEVCIES' in os.environ:
		sys.exit(0)
	if selected_gpus is None:
		mem_trace = []
		utility_trace = []
		for i in range(5): # sample 5 times
			info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
			mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
			utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
			mem_trace.append(mem)
			utility_trace.append(utility)
			time.sleep(0.1)
		mem = np.mean(mem_trace, axis=0)
		utility = np.mean(utility_trace, axis=0)
		assert(len(mem) == len(utility))
		nGPU = len(utility)
		ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

		if len(ideal_gpus) < num_gpu:
			print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
			sys.exit(0)
		else:
			selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
	else:
		selected_gpus = selected_gpus.split(',')

	print("Setting GPU: {}".format(selected_gpus))
	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
	return selected_gpus

def expand_user(path):
	return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
	from collections import OrderedDict
	import torch
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	if old_file and os.path.exists(expand_user(old_file)):
		if verbose:
			print("Removing old model {}".format(expand_user(old_file)))
		os.remove(expand_user(old_file))
	if verbose:
		print("Saving model to {}".format(expand_user(new_file)))

	state_dict = OrderedDict()
	for k, v in model.state_dict().items():
		if v.is_cuda:
			v = v.cpu()
		state_dict[k] = v
	torch.save(state_dict, expand_user(new_file))
 

def load_lmdb(lmdb_file, n_records=None):
	import lmdb
	import numpy as np
	lmdb_file = expand_user(lmdb_file)
	if os.path.exists(lmdb_file):
		data = []
		env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
		with env.begin() as txn:
			cursor = txn.cursor()
			begin_st = time.time()
			print("Loading lmdb file {} into memory".format(lmdb_file))
			for key, value in cursor:
				_, target, _ = key.decode('ascii').split(':')
				target = int(target)
				img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
				data.append((img, target))
				if n_records is not None and len(data) >= n_records:
					break
		env.close()
		print("=> Done ({:.4f} s)".format(time.time() - begin_st))
		return data
	else:
		print("Not found lmdb file".format(lmdb_file))

def str2img(str_b):
	return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
	return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
	m = hashlib.md5()
	m.update(s)
	return m.hexdigest()

def eval_model(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
	import tqdm
	import torch
	from torch import nn
	from torch.autograd import Variable

import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
	'''
	Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
	Input:
	  A: Nxm numpy array of corresponding points
	  B: Nxm numpy array of corresponding points
	Returns:
	  T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
	  R: mxm rotation matrix
	  t: mx1 translation vector
	'''

	assert A.shape == B.shape

	# get number of dimensions
	m = A.shape[1]

	# translate points to their centroids
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	AA = A - centroid_A
	BB = B - centroid_B

	# rotation matrix
	H = np.dot(AA.T, BB)
	U, S, Vt = np.linalg.svd(H)
	R = np.dot(Vt.T, U.T)

	# special reflection case
	if np.linalg.det(R) < 0:
	   Vt[m-1,:] *= -1
	   R = np.dot(Vt.T, U.T)

	# translation
	t = centroid_B.T - np.dot(R,centroid_A.T)

	# homogeneous transformation
	T = np.identity(m+1)
	T[:m, :m] = R
	T[:m, m] = t

	return T, R, t


def nearest_neighbor(src, dst):
	'''
	Find the nearest (Euclidean) neighbor in dst for each point in src
	Input:
		src: Nxm array of points
		dst: Nxm array of points
	Output:
		distances: Euclidean distances of the nearest neighbor
		indices: dst indices of the nearest neighbor
	'''

	assert src.shape == dst.shape

	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(dst)
	distances, indices = neigh.kneighbors(src, return_distance=True)
	return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
	'''
	The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
	Input:
		A: Nxm numpy array of source mD points
		B: Nxm numpy array of destination mD point
		init_pose: (m+1)x(m+1) homogeneous transformation
		max_iterations: exit algorithm after max_iterations
		tolerance: convergence criteria
	Output:
		T: final homogeneous transformation that maps A on to B
		distances: Euclidean distances (errors) of the nearest neighbor
		i: number of iterations to converge
	'''

	assert A.shape == B.shape

	# get number of dimensions
	m = A.shape[1]

	# make points homogeneous, copy them to maintain the originals
	src = np.ones((m+1,A.shape[0]))
	dst = np.ones((m+1,B.shape[0]))
	src[:m,:] = np.copy(A.T)
	dst[:m,:] = np.copy(B.T)

	# apply the initial pose estimation
	if init_pose is not None:
		src = np.dot(init_pose, src)

	prev_error = 0

	for i in range(max_iterations):
		# find the nearest neighbors between the current source and destination points
		distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

		# compute the transformation between the current source and nearest destination points
		T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

		# update the current source
		src = np.dot(T, src)

		# check error
		mean_error = np.mean(distances)
		if np.abs(prev_error - mean_error) < tolerance:
			break
		prev_error = mean_error

	# calculate final transformation
	T,_,_ = best_fit_transform(A, src[:m,:].T)

	return T, distances, i

def load_state_dict(model, model_root):
	from torch import nn
	import torch
	import re
	from collections import OrderedDict
	own_state_old = model.state_dict()
	own_state = OrderedDict() # remove all 'group' string
	for k, v in own_state_old.items():
		#print(k)
		#k = re.sub('group\d+\.', '', k)
		own_state[k] = v
		#print(k)
	state = torch.load(model_root)
	state_dict = state['dict']
	state_dict = {k: v for k, v in state_dict.items() if k in own_state} #filter out unnecessary keys
	#print(state_dict.items())
	for name, param in state_dict.items():
		if name not in own_state:
			print(own_state.keys())
			raise KeyError('unexpected key "{}" in state_dict'
						   .format(name))
		if isinstance(param, nn.Parameter):
			# backwards compatibility for serialized parameters
			param = param.data
		own_state[name].copy_(param)
	

	#bfm=np.load("../propressing/bfma.npz")
	#own_state['A_BFMS.weight'].copy_(torch.Tensor(bfm['w_shape']))
	#own_state['A_BFMS.bias'].copy_(torch.Tensor(bfm['mu_shape'].reshape(-1)))

	# own_state['BFME.weight'].copy_(torch.Tensor(bfm['w_expression']))
	# own_state['BFME.bias'].copy_(torch.Tensor(bfm['mu_expression'].reshape(-1)))

	# missing = set(own_state.keys()) - set(state_dict.keys())
	# if len(missing) > 0:
	#	 raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def load_state_dict_bfm(model, model_root):
	import torch
	from torch import nn
	from collections import OrderedDict
	own_state = model.state_dict()
	#own_state = OrderedDict()
	#print(own_state.items())
	bfm=np.load("../propressing/bfma.npz")
	own_state['A_BFMS.weight'].copy_(torch.Tensor(bfm['w_shape']))
	own_state['A_BFMS.bias'].copy_(torch.Tensor(bfm['mu_shape'].reshape(-1)))
	print(model.state_dict())
	# own_state['BFME.weight'].copy_(torch.Tensor(bfm['w_expression']))
	# own_state['BFME.bias'].copy_(torch.Tensor(bfm['mu_expression'].reshape(-1)))



	
	