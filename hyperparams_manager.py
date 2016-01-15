
from constants import *
from environments import *
from training_settings import *

import cPickle as cP
import time

import numpy as np
import adjspecies
import pprint	

import pdb


class Hyperparams_Manager():
	def __init__(self):
		self.dict = {} #key:adjspecies, value: setting_dict
		self.dict_name2str = {} #key: ajdspecies, value: str(setting_dict.values())
		self.dict_str2name = {} # reverse of above.

	def check_and_save(self, setting_dict):
		if not has_setting(setting_dict):
			self.save_new_setting(setting_dict)
		else:
			print 'This setting dictionary is already stored.'

	def get_name(self, setting_dict):
		'''return new name if it is new.
		return previously used name if it is the same again.
		'''
		# if self.has_setting(setting_dict):
		# 	print 'return once used name:%s' %  self.dict_str2name[dict2str(setting_dict)]
		# 	return self.dict_str2name[dict2str(setting_dict)]
		# else:
		# 	return self.pick_new_name()
		return self.pick_new_name()

	def pick_new_name(self):
		new_name = adjspecies.random_adjspecies(sep='_', maxlen=10, prevent_stutter=True)
		while new_name in self.dict:
			new_name = adjspecies.random_adjspecies(sep='_', maxlen=10, prevent_stutter=True)
		return new_name

	def save_new_setting(self, setting_dict):	
		new_name = self.pick_new_name()	
		self.dict[new_name] = setting_dict
		long_name = dict2str(setting_dict)
		self.dict_name2str[new_name]= long_name
		self.dict_str2name[long_name] = new_name
		cP.dump(self, open(PATH_DATA + FILE_DICT["hyperparam_manager"], 'w'))

	def list_setting_names(self):
		pprint.pprint(self.dict.keys())

	def list_settings(self):
		pprint.pprint(self.dict)

	def has_setting(self, setting_dict):
		return dict2str(setting_dict) in self.dict_str2name

	def write_setting_as_texts(self, path_to_save, setting_dict): # TODO : move this to hyperparams()
		from datetime import datetime
		timeinfo = datetime.now().strftime('%Y-%m-%d-%Hh%Mm')
		f = open(path_to_save + timeinfo + '.time', 'w')
		f.close()
		for key in setting_dict:
			with open(path_to_save+ 'a_'+str(key)+ '_' + str(setting_dict[key])+'.txt', 'w') as f:
				pass
		return

	def print_setting(self, setting_dict):
		print '-'*60
		for key in setting_dict:
			print( ' * ' + str(key)+ ': ' + str(setting_dict[key]))
		print '-'*60
		return