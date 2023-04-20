
from strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
								LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
								KMeansSampling, KMeansSamplingGPU, KCenterGreedy, KCenterGreedyPCA, BALDDropout,  \
								AdversarialBIM, AdversarialDeepFool, VarRatio, MeanSTD, BadgeSampling, CEALSampling, \
								LossPredictionLoss, VAAL, WAAL, MetaSampling
from torchvision import transforms
import sys
import os
import numpy as np
import math
import torch
import typing as ty
import json
from pathlib import Path

#strategy

def get_strategy(STRATEGY_NAME, dataset):
	if STRATEGY_NAME == 'RandomSampling':
		return RandomSampling(dataset)
	elif STRATEGY_NAME == 'LeastConfidence':
		return LeastConfidence(dataset)
	elif STRATEGY_NAME == 'MarginSampling':
		return MarginSampling(dataset)
	elif STRATEGY_NAME == 'EntropySampling':
		return EntropySampling(dataset)
	elif STRATEGY_NAME == 'LeastConfidenceDropout':
		return LeastConfidenceDropout(dataset)
	elif STRATEGY_NAME == 'MarginSamplingDropout':
		return MarginSamplingDropout(dataset)
	elif STRATEGY_NAME == 'EntropySamplingDropout':
		return EntropySamplingDropout(dataset)
	elif STRATEGY_NAME == 'KMeansSampling':   # embedding
		return KMeansSampling(dataset)
	elif STRATEGY_NAME == 'KMeansSamplingGPU': # embedding
		return KMeansSamplingGPU(dataset)
	elif STRATEGY_NAME == 'KCenterGreedy':  # embedding
		return KCenterGreedy(dataset)
	elif STRATEGY_NAME == 'KCenterGreedyPCA':  # embedding
		return KCenterGreedyPCA(dataset)
	elif STRATEGY_NAME == 'BALDDropout':
		return BALDDropout(dataset)
	elif STRATEGY_NAME == 'VarRatio':
		return VarRatio(dataset)
	elif STRATEGY_NAME == 'MeanSTD':
		return MeanSTD(dataset)
	elif STRATEGY_NAME == 'BadgeSampling':  # embedding
		return BadgeSampling(dataset)
	elif STRATEGY_NAME == 'LossPredictionLoss':   #lpl
		return LossPredictionLoss(dataset)
	elif STRATEGY_NAME == 'AdversarialBIM':  # require embedding and adversarial
		return AdversarialBIM(dataset)
	elif STRATEGY_NAME == 'AdversarialDeepFool':
		return AdversarialDeepFool(dataset)
	# elif 'CEALSampling' in STRATEGY_NAME:
	# 	return CEALSampling(dataset)
	# elif STRATEGY_NAME == 'VAAL':
	# 	net_vae,net_disc = get_net_vae(args_task['name'])
	# 	handler_joint = get_handler_joint(args_task['name'])
	# 	return VAAL(dataset, net, net_vae = net_vae, net_dis = net_disc, handler_joint = handler_joint)
	elif STRATEGY_NAME == 'WAAL':
		return WAAL(dataset)
	elif STRATEGY_NAME == 'MetaSampling':
		return MetaSampling(dataset)
	else:
		raise NotImplementedError

#other stuffs

# logger
class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)

def get_aubc(quota, bsize, resseq):
	# it is equal to use np.trapz for calculation
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum

def load_json(path: ty.Union[Path, str]) -> ty.Any:
    return json.loads(Path(path).read_text())

def dump_json(x: ty.Any, path: ty.Union[Path, str], *args, **kwargs) -> None:
    Path(path).write_text(json.dumps(x, *args, **kwargs) + '\n')
