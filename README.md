Introduction
============

This porject is porting our original code in Theano to Tensorflow. Still in very early stage and under heavy development.


INTRODUCTION
============

Implementation of Neural Language Correction (http://arxiv.org/abs/1603.09727) on Tensorflow

DEPENDENCIES
============

Tensorflow 0.9


TRAINING
========

To train character level model (default):

   $ python train.py


To train word level model:

   $ python train.py --tokenizer WORD
# hs
# 问题一：
# 运行之后出现socket error 看错误提示发现，程序先从斯坦福的地址下载要处理的语料，然后处理语料，socket error 原因就是链接错误，
#　尝试科大ＶＰＮ之后还是socket error
# 方案：　暂时搁置，先了解清除处理流程，语料下载之后是怎么处理的　　
# 后续思考：　下载下来的语料形式？　语言本身还是直接Ｗ２ｖ的结果，看处理过程应该是语句本身，然后做embedding 处理（语句转向量）
# 问题二：
# train 依赖于　nlc_model 和　nlc_data
# nlc_model 依赖于　nlc_data
# nlc_data 检测训练数据是否下载，未下载则下载并处理
# 要点：　核心时nlc_model 文件，　不是train ，train文件基本没啥看不懂的，核心在域掉用的nlc_model 中的函数
# 要点：　util 处理分割符即tokenizer，　decoder 做语句分割　，nlc_model建立模型类，　nlc_data　获得训练数据, train启动训练处理　
# 
INTERACTIVE DECODING
====================

   $ python decode.py



