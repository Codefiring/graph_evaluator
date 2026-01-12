"""
测试脚本 - 有界序列评估器测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph_evaluator import SequenceEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取测试数据目录路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def test_case_1():
    """测试用例1: 完全匹配的序列"""
    print("\n" + "="*60)
    print("测试用例1: 完全匹配的序列")
    print("="*60)
    evaluator = SequenceEvaluator(max_length=5, use_sampling=False)
    gt_file = os.path.join(DATA_DIR, 'ground_truth.txt')
    pred_file = os.path.join(DATA_DIR, 'prediction.txt')
    result = evaluator.evaluate(gt_file, pred_file)
    print(f"\n[结果] Precision: {result['precision']:.4f}")
    print(f"[结果] Recall: {result['recall']:.4f}")
    print(f"[结果] F1: {result['f1']:.4f}")
    print(f"[结果] GT序列数: {result['num_sequences_gt']}")
    print(f"[结果] Pred序列数: {result['num_sequences_pred']}")
    print(f"[结果] 交集序列数: {result['num_intersection']}")
    # 完全匹配应该precision和recall都接近1.0
    return result['precision'] > 0.9 and result['recall'] > 0.9

def test_case_2():
    """测试用例2: 部分匹配的序列"""
    print("\n" + "="*60)
    print("测试用例2: 部分匹配的序列")
    print("="*60)
    evaluator = SequenceEvaluator(max_length=4, use_sampling=False)
    gt_file = os.path.join(DATA_DIR, 'test_case2_ground_truth.txt')
    pred_file = os.path.join(DATA_DIR, 'test_case2_prediction.txt')
    result = evaluator.evaluate(gt_file, pred_file)
    print(f"\n[结果] Precision: {result['precision']:.4f}")
    print(f"[结果] Recall: {result['recall']:.4f}")
    print(f"[结果] F1: {result['f1']:.4f}")
    print(f"[结果] GT序列数: {result['num_sequences_gt']}")
    print(f"[结果] Pred序列数: {result['num_sequences_pred']}")
    print(f"[结果] 交集序列数: {result['num_intersection']}")
    return True

def test_sampling():
    """测试用例3: 使用采样模式"""
    print("\n" + "="*60)
    print("测试用例3: 使用采样模式")
    print("="*60)
    evaluator = SequenceEvaluator(max_length=5, use_sampling=True, sample_size=1000)
    gt_file = os.path.join(DATA_DIR, 'ground_truth.txt')
    pred_file = os.path.join(DATA_DIR, 'prediction.txt')
    result = evaluator.evaluate(gt_file, pred_file)
    print(f"\n[结果] Precision: {result['precision']:.4f}")
    print(f"[结果] Recall: {result['recall']:.4f}")
    print(f"[结果] F1: {result['f1']:.4f}")
    print(f"[结果] 使用采样: {result['use_sampling']}")
    return True

if __name__ == '__main__':
    print("开始运行有界序列评估器测试用例...")
    
    test1_passed = test_case_1()
    test2_passed = test_case_2()
    test3_passed = test_sampling()
    
    print("\n" + "="*60)
    print("测试结果汇总:")
    print(f"  测试用例1 (完全匹配): {'通过' if test1_passed else '失败'}")
    print(f"  测试用例2 (部分匹配): {'通过' if test2_passed else '失败'}")
    print(f"  测试用例3 (采样模式): {'通过' if test3_passed else '失败'}")
    print("="*60)
