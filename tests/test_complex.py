"""
测试复杂用例
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph_evaluator import GraphEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取测试数据目录路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def test_complex_case():
    """测试复杂用例: Ground Truth 17条边, Prediction 24条边"""
    print("\n" + "="*60)
    print("复杂测试用例")
    print("="*60)
    evaluator = GraphEvaluator()
    gt_file = os.path.join(DATA_DIR, 'complex_ground_truth.txt')
    pred_file = os.path.join(DATA_DIR, 'complex_prediction.txt')
    result = evaluator.evaluate(gt_file, pred_file)
    
    print("\n" + "="*60)
    print("评估结果摘要:")
    print(f"  总编辑距离: {result['total_distance']}")
    print(f"  边相似度: {result['edge_similarity']:.4f}")
    print(f"  共同边数: {result['edges_common']}")
    print(f"  需要添加的边数: {result['edges_to_add']}")
    print(f"  需要删除的边数: {result['edges_to_delete']}")
    print("="*60)
    
    if result['edges_to_add'] > 0:
        print(f"\n需要添加的边详情 ({len(result['edges_to_add_details'])}条):")
        for edge in result['edges_to_add_details'][:10]:  # 只显示前10条
            print(f"  {edge}")
        if len(result['edges_to_add_details']) > 10:
            print(f"  ... 还有 {len(result['edges_to_add_details']) - 10} 条")
    
    if result['edges_to_delete'] > 0:
        print(f"\n需要删除的边详情 ({len(result['edges_to_delete_details'])}条):")
        for edge in result['edges_to_delete_details'][:10]:  # 只显示前10条
            print(f"  {edge}")
        if len(result['edges_to_delete_details']) > 10:
            print(f"  ... 还有 {len(result['edges_to_delete_details']) - 10} 条")
    
    print(f"\n节点映射关系:")
    for pred_node, gt_node in list(result['node_mapping'].items())[:10]:
        print(f"  {pred_node} -> {gt_node}")
    if len(result['node_mapping']) > 10:
        print(f"  ... 还有 {len(result['node_mapping']) - 10} 个节点映射")

if __name__ == '__main__':
    test_complex_case()
