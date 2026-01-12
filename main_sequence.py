"""
主入口文件 - 有界序列评估工具
用于评估预测状态机与ground truth状态机之间可接受的ioctl序列集合的相似度
"""

from graph_evaluator import SequenceEvaluator
import sys
import os

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("使用方法: python main_sequence.py <ground_truth_file> <prediction_file> [max_length] [--sampling]")
        print("参数说明:")
        print("  ground_truth_file: Ground Truth文件路径")
        print("  prediction_file: 预测结果文件路径")
        print("  max_length: 最大序列长度k（可选，默认5）")
        print("  --sampling: 使用采样模式（可选，用于分支过多的情况）")
        print("\n示例:")
        print("  python main_sequence.py tests/data/ground_truth.txt tests/data/prediction.txt")
        print("  python main_sequence.py tests/data/ground_truth.txt tests/data/prediction.txt 6")
        print("  python main_sequence.py tests/data/ground_truth.txt tests/data/prediction.txt 5 --sampling")
        sys.exit(1)
    
    ground_truth_file = sys.argv[1]
    prediction_file = sys.argv[2]
    
    # 解析可选参数
    max_length = 5
    use_sampling = False
    
    if len(sys.argv) >= 4:
        try:
            max_length = int(sys.argv[3])
        except ValueError:
            if sys.argv[3] == '--sampling':
                use_sampling = True
            else:
                print(f"错误: 无效的最大长度参数: {sys.argv[3]}")
                sys.exit(1)
    
    if '--sampling' in sys.argv:
        use_sampling = True
    
    if not os.path.exists(ground_truth_file):
        print(f"错误: Ground Truth文件不存在: {ground_truth_file}")
        sys.exit(1)
    
    if not os.path.exists(prediction_file):
        print(f"错误: 预测结果文件不存在: {prediction_file}")
        sys.exit(1)
    
    # 创建评估器
    evaluator = SequenceEvaluator(
        max_length=max_length,
        use_sampling=use_sampling
    )
    
    # 评估
    result = evaluator.evaluate(ground_truth_file, prediction_file)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果汇总:")
    print("="*60)
    print(f"序列级 Precision: {result['precision']:.4f}")
    print(f"序列级 Recall: {result['recall']:.4f}")
    print(f"F1分数: {result['f1']:.4f}")
    print(f"\nGround Truth序列数: {result['num_sequences_gt']}")
    print(f"Prediction序列数: {result['num_sequences_pred']}")
    print(f"交集序列数: {result['num_intersection']}")
    print(f"最大序列长度: {result['max_length']}")
    print(f"使用采样: {result['use_sampling']}")
    print("="*60)
    print("\n详细日志已保存到 logs/sequence_evaluation.log")

if __name__ == '__main__':
    main()
