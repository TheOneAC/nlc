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
# 要点：　核心是decode 文件中的fix_sent函数


#　目前可深究小问题：
# socket　链接下载训练数据
# train 中涉及日志处理，py 的log日志包可以熟悉并做下笔记
#　nlc_data 　中　正则表达式处理部分未细究可查看py的正则并深究并作下正则解析过程记录
# decode 编码评分规则可以进一步理解下
#主要工作范围缩小至　nlc_model 类定义，　nlc_data 的正则处理
# 建议工作方法：　先在看不懂的位置加问题，然后自己尝试解答，或者让他人解答，能解答一定是代码逻辑很清楚了
# 如果看不懂，就在相应位置加问题



INTERACTIVE DECODING
====================

   $ python decode.py



