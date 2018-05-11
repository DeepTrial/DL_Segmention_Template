"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""
import random,numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

from perception.bases.trainer_base import TrainerBase
from configs.utils.utils import genMasks
from configs.utils.img_utils import img_process

class SegmentionTrainer(TrainerBase):
	def __init__(self,model,data,config):
		super(SegmentionTrainer, self).__init__(model, data, config)
		self.model=model
		self.data=data
		self.config=config
		self.callbacks=[]
		self.init_callbacks()

	def init_callbacks(self):
		self.callbacks.append(
			ModelCheckpoint(
				filepath=self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5',
		        verbose=1,
		        monitor='val_loss',
		        mode='auto',
		        save_best_only=True
			)
		)

		self.callbacks.append(
			TensorBoard(
				log_dir=self.config.checkpoint,
				write_images=True,
				write_graph=True,
			)
		)

	def train(self):
		gen=DataGenerator(self.data,self.config)
		hist = self.model.fit_generator(gen.train_gen(),
		    epochs=self.config.epochs,
		    steps_per_epoch=self.config.subsample * self.config.total_train / self.config.batch_size,
		    verbose=1,
		    callbacks=self.callbacks,
		    validation_data=gen.val_gen(),
			validation_steps=int(self.config.subsample * self.config.total_val / self.config.batch_size)
		)
		self.model.save_weights(self.config.hdf5_path+self.config.exp_name+'_last_weights.h5', overwrite=True)



class DataGenerator():
	"""
	图片取块
	"""
	def __init__(self,data,config):
		self.train_img=img_process(data[0])
		self.train_gt=data[1]/255.
		self.val_img=img_process(data[2])
		self.val_gt=data[3]/255.
		self.config=config

	def _CenterSampler(self,attnlist):
		"""
		围绕目标区域采样
		:param attnlist:  目标区域坐标
		:return: 采样的坐标
		"""

		label=0
		t = attnlist[label]
		cid = random.randint(0, t[0].shape[0] - 1)
		i_center = t[0][cid]
		y_center = t[1][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))
		x_center = t[2][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))

		if y_center < self.config.patch_width / 2:
			y_center = self.config.patch_width / 2
		elif y_center > self.config.height - self.config.patch_width / 2:
			y_center = self.config.height - self.config.patch_width / 2

		if x_center < self.config.patch_width / 2:
			x_center = self.config.patch_width / 2
		elif x_center > self.config.width - self.config.patch_width / 2:
			x_center = self.config.width - self.config.patch_width / 2

		return i_center, x_center, y_center

	def _genDef(self,train_imgs,train_masks,attnlist):
		"""
		图片取块生成器模板
		:param train_imgs: 原始图
		:param train_masks:  原始图groundtruth
		:param attnlist:  目标区域list
		:return:  取出的训练样本
		"""
		while 1:
			for t in range(int(self.config.subsample * self.config.total_train / self.config.batch_size)):
				X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,3])
				Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])
				for j in range(self.config.batch_size):
					[i_center, x_center, y_center] = self._CenterSampler(attnlist)
					patch = train_imgs[i_center, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2),:]
					patch_mask = train_masks[i_center, :, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2)]
					X[j, :, :, :] = patch
					Y[j, :, :] = genMasks(np.reshape(patch_mask, [1, self.config.seg_num, self.config.patch_height, self.config.patch_width]),self.config.seg_num)
				yield (X, Y)

	def train_gen(self):
		"""
		训练样本生成器
		"""
		attnlist=[np.where(self.train_gt[:,0,:,:]==np.max(self.train_gt[:,0,:,:]))]
		return self._genDef(self.train_img,self.train_gt,attnlist)

	def val_gen(self):
		"""
		验证样本生成器
		"""
		attnlist = [np.where(self.val_gt[:, 0, :, :] == np.max(self.val_gt[:, 0, :, :]))]
		return self._genDef(self.val_img, self.val_gt, attnlist)