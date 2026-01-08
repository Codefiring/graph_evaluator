"""
测试节点名称差异被完全忽略
"""

from graph_evaluator import GraphEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_node_name_ignoring():
    """测试节点名称完全不同的情况"""
    print("\n" + "="*60)
    print("测试：节点名称完全不同的图")
    print("="*60)
    
    # 创建测试数据
    gt_content = """S1, operator1, S2
S2, operator2, S3
S3, operator1, S1"""
    
    pred_content = """X, operator1, Y
Y, operator2, Z
Z, operator1, X"""
    
    with open('test_gt.txt', 'w', encoding='utf-8') as f:
        f.write(gt_content)
    
    with open('test_pred.txt', 'w', encoding='utf-8') as f:
        f.write(pred_content)
    
    evaluator = GraphEvaluator()
    result = evaluator.evaluate('test_gt.txt', 'test_pred.txt')
    
    print("\n" + "="*60)
    print("测试结果:")
    print(f"  边相似度: {result['edge_similarity']:.4f} (应该接近1.0)")
    print(f"  总编辑距离: {result['total_distance']} (应该为0.0)")
    print(f"  共同边数: {result['edges_common']} (应该为3)")
    print(f"  节点映射:")
    for pred_node, gt_node in result['node_mapping'].items():
        print(f"    {pred_node} -> {gt_node}")
    print("="*60)
    
    # 验证结果
    assert result['edge_similarity'] == 1.0, f"边相似度应该是1.0，实际是{result['edge_similarity']}"
    assert result['total_distance'] == 0.0, f"编辑距离应该是0.0，实际是{result['total_distance']}"
    assert result['edges_common'] == 3, f"共同边数应该是3，实际是{result['edges_common']}"
    
    print("\n[PASS] 测试通过：节点名称差异被完全忽略！")

if __name__ == '__main__':
    test_node_name_ignoring()

