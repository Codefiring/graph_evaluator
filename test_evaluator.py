"""
测试脚本 - 运行多个测试用例
"""

from graph_evaluator import GraphEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_case_1():
    """测试用例1: 完全匹配（节点名称不同但结构相同）"""
    print("\n" + "="*60)
    print("测试用例1: 完全匹配")
    print("="*60)
    evaluator = GraphEvaluator()
    result = evaluator.evaluate('ground_truth.txt', 'prediction.txt')
    print(f"\n[PASS] 边相似度: {result['edge_similarity']:.4f}")
    print(f"[PASS] 总编辑距离: {result['total_distance']}")
    return result['edge_similarity'] == 1.0

def test_case_2():
    """测试用例2: 部分匹配（有额外的边）"""
    print("\n" + "="*60)
    print("测试用例2: 部分匹配（预测结果有额外边）")
    print("="*60)
    evaluator = GraphEvaluator()
    result = evaluator.evaluate('test_case2_ground_truth.txt', 'test_case2_prediction.txt')
    print(f"\n[PASS] 边相似度: {result['edge_similarity']:.4f}")
    print(f"[PASS] 总编辑距离: {result['total_distance']}")
    print(f"[PASS] 需要删除的边数: {result['edges_to_delete']}")
    print(f"[PASS] 需要添加的边数: {result['edges_to_add']}")
    return result['edge_similarity'] < 1.0 and result['edges_to_add'] > 0

if __name__ == '__main__':
    print("开始运行测试用例...")
    
    test1_passed = test_case_1()
    test2_passed = test_case_2()
    
    print("\n" + "="*60)
    print("测试结果汇总:")
    print(f"  测试用例1 (完全匹配): {'通过' if test1_passed else '失败'}")
    print(f"  测试用例2 (部分匹配): {'通过' if test2_passed else '失败'}")
    print("="*60)

