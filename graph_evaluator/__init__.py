"""
图编辑距离评估算法包
用于评估预测图与ground truth图之间的相似度
支持忽略节点名称差异，只关注图的结构和边标签
"""

from .evaluator import GraphEvaluator
from .sequence_evaluator import SequenceEvaluator

__version__ = '1.0.0'
__all__ = ['GraphEvaluator', 'SequenceEvaluator']
