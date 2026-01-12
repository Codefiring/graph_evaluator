"""
主入口文件 - 图编辑距离评估工具
"""

from graph_evaluator import GraphEvaluator
import sys
import os

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("使用方法: python main.py <ground_truth_file> <prediction_file>")
        print("示例: python main.py tests/data/ground_truth.txt tests/data/prediction.txt")
        sys.exit(1)
    
    ground_truth_file = sys.argv[1]
    prediction_file = sys.argv[2]
    
    if not os.path.exists(ground_truth_file):
        print(f"错误: Ground Truth文件不存在: {ground_truth_file}")
        sys.exit(1)
    
    if not os.path.exists(prediction_file):
        print(f"错误: 预测结果文件不存在: {prediction_file}")
        sys.exit(1)
    
    evaluator = GraphEvaluator()
    
    # 评估
    result = evaluator.evaluate(ground_truth_file, prediction_file)
    
    print("\n评估完成！详细日志已保存到 logs/graph_evaluation.log")
    print(f"边相似度: {result['edge_similarity']:.4f}")

if __name__ == '__main__':
    main()
