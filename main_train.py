
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10

"""
import os
from experiments.data_loaders.standard_loader import DataLoader
# from infers.simple_mnist_infer import SimpleMnistInfer
from perception.models.segmention_model import SegmentionModel
from perception.trainers.segmention_trainer import SegmentionTrainer
from configs.utils.config_utils import process_config
import numpy as np


def main_train():
    """
    训练模型

    :return:
    """
    print('[INFO] 解析配置...')

    config = None

    try:
        config = process_config('configs/segmention_config.json')
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        exit(0)
    # np.random.seed(47)  # 固定随机数

    print('[INFO] 加载数据...')
    dataloader = DataLoader(config=config)
    dataloader.prepare_dataset()

    train_imgs,train_gt=dataloader.get_train_data()
    val_imgs,val_gt=dataloader.get_val_data()

    print('[INFO] 构造网络...')
    model = SegmentionModel(config=config)
    #
    print('[INFO] 训练网络...')
    trainer = SegmentionTrainer(
         model=model.model,
         data=[train_imgs,train_gt,val_imgs,val_gt],
         config=config)
    trainer.train()
    print('[INFO] 训练完成...')



if __name__ == '__main__':
    main_train()
    # test_main()
