
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

from perception.infers.segmention_infer import SegmentionInfer
from configs.utils.config_utils import process_config


def main_test():
    print('[INFO] 解析配置...')
    config = None

    try:
        config = process_config('configs/segmention_config.json')
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        exit(0)


    print('[INFO] 预测数据...')
    infer = SegmentionInfer( config)
    infer.predict()

    print('[INFO] 预测完成...')


if __name__ == '__main__':
    main_test()
