"""
批量评估脚本
支持通过配置文件批量评估多个项目
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from graph_evaluator import SequenceEvaluator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Dict:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {e}")
        sys.exit(1)


def evaluate_project(project: Dict, evaluator: SequenceEvaluator) -> Dict[str, Any]:
    """
    评估单个项目
    
    Args:
        project: 项目配置字典
        evaluator: 评估器实例
        
    Returns:
        评估结果字典
    """
    project_id = project.get('id', 'unknown')
    project_name = project.get('name', project_id)
    gt_file = project.get('ground_truth_file')
    pred_file = project.get('prediction_file')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始评估项目: {project_name} ({project_id})")
    logger.info(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(gt_file):
        logger.error(f"Ground Truth文件不存在: {gt_file}")
        return {
            'project_id': project_id,
            'project_name': project_name,
            'status': 'error',
            'error_message': f'Ground Truth文件不存在: {gt_file}'
        }
    
    if not os.path.exists(pred_file):
        logger.error(f"Prediction文件不存在: {pred_file}")
        return {
            'project_id': project_id,
            'project_name': project_name,
            'status': 'error',
            'error_message': f'Prediction文件不存在: {pred_file}'
        }
    
    try:
        # 执行评估
        result = evaluator.evaluate(gt_file, pred_file)
        
        # 添加项目信息
        result['project_id'] = project_id
        result['project_name'] = project_name
        result['status'] = 'success'
        result['ground_truth_file'] = gt_file
        result['prediction_file'] = pred_file
        result['evaluation_time'] = datetime.now().isoformat()
        
        logger.info(f"项目 {project_name} 评估完成")
        return result
        
    except Exception as e:
        logger.error(f"评估项目 {project_name} 时出错: {e}", exc_info=True)
        return {
            'project_id': project_id,
            'project_name': project_name,
            'status': 'error',
            'error_message': str(e),
            'ground_truth_file': gt_file,
            'prediction_file': pred_file,
            'evaluation_time': datetime.now().isoformat()
        }


def save_results_json(results: List[Dict], output_file: str):
    """
    保存结果为JSON格式
    
    Args:
        results: 评估结果列表
        output_file: 输出文件路径
    """
    output_data = {
        'evaluation_time': datetime.now().isoformat(),
        'total_projects': len(results),
        'successful_projects': sum(1 for r in results if r.get('status') == 'success'),
        'failed_projects': sum(1 for r in results if r.get('status') == 'error'),
        'results': results
    }
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON结果已保存到: {output_file}")


def save_results_csv(results: List[Dict], output_file: str):
    """
    保存结果为CSV格式
    
    Args:
        results: 评估结果列表
        output_file: 输出文件路径
    """
    import csv
    
    # 定义CSV列
    fieldnames = [
        'project_id', 'project_name', 'status',
        'precision', 'recall', 'f1', 'jaccard',
        'num_sequences_gt', 'num_sequences_pred', 'num_intersection',
        'num_only_in_pred', 'num_only_in_gt',
        'gt_num_nodes', 'gt_num_edges',
        'pred_num_nodes', 'pred_num_edges',
        'coverage_gt', 'coverage_pred',
        'gt_avg_seq_length', 'pred_avg_seq_length',
        'gt_min_seq_length', 'gt_max_seq_length',
        'pred_min_seq_length', 'pred_max_seq_length',
        'max_length', 'use_sampling',
        'evaluation_time', 'error_message'
    ]
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'project_id': result.get('project_id', ''),
                'project_name': result.get('project_name', ''),
                'status': result.get('status', ''),
                'precision': result.get('precision', ''),
                'recall': result.get('recall', ''),
                'f1': result.get('f1', ''),
                'jaccard': result.get('jaccard', ''),
                'num_sequences_gt': result.get('num_sequences_gt', ''),
                'num_sequences_pred': result.get('num_sequences_pred', ''),
                'num_intersection': result.get('num_intersection', ''),
                'num_only_in_pred': result.get('num_only_in_pred', ''),
                'num_only_in_gt': result.get('num_only_in_gt', ''),
                'gt_num_nodes': result.get('gt_num_nodes', ''),
                'gt_num_edges': result.get('gt_num_edges', ''),
                'pred_num_nodes': result.get('pred_num_nodes', ''),
                'pred_num_edges': result.get('pred_num_edges', ''),
                'coverage_gt': result.get('coverage_gt', ''),
                'coverage_pred': result.get('coverage_pred', ''),
                'gt_avg_seq_length': result.get('gt_sequence_stats', {}).get('avg_length', ''),
                'pred_avg_seq_length': result.get('pred_sequence_stats', {}).get('avg_length', ''),
                'gt_min_seq_length': result.get('gt_sequence_stats', {}).get('min_length', ''),
                'gt_max_seq_length': result.get('gt_sequence_stats', {}).get('max_length', ''),
                'pred_min_seq_length': result.get('pred_sequence_stats', {}).get('min_length', ''),
                'pred_max_seq_length': result.get('pred_sequence_stats', {}).get('max_length', ''),
                'max_length': result.get('max_length', ''),
                'use_sampling': result.get('use_sampling', ''),
                'evaluation_time': result.get('evaluation_time', ''),
                'error_message': result.get('error_message', '')
            }
            writer.writerow(row)
    
    logger.info(f"CSV结果已保存到: {output_file}")


def print_summary(results: List[Dict]):
    """
    打印评估结果汇总
    
    Args:
        results: 评估结果列表
    """
    print("\n" + "="*80)
    print("批量评估结果汇总")
    print("="*80)
    
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'error']
    
    print(f"\n总项目数: {len(results)}")
    print(f"成功评估: {len(successful_results)}")
    print(f"失败评估: {len(failed_results)}")
    
    if successful_results:
        print("\n" + "-"*80)
        print("成功评估的项目详情:")
        print("-"*80)
        print(f"{'项目ID':<15} {'项目名称':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Jaccard':<12}")
        print("-"*80)
        
        for result in successful_results:
            print(f"{result.get('project_id', ''):<15} "
                  f"{result.get('project_name', ''):<20} "
                  f"{result.get('precision', 0):<12.4f} "
                  f"{result.get('recall', 0):<12.4f} "
                  f"{result.get('f1', 0):<12.4f} "
                  f"{result.get('jaccard', 0):<12.4f}")
        
        # 计算平均值
        avg_precision = sum(r.get('precision', 0) for r in successful_results) / len(successful_results)
        avg_recall = sum(r.get('recall', 0) for r in successful_results) / len(successful_results)
        avg_f1 = sum(r.get('f1', 0) for r in successful_results) / len(successful_results)
        avg_jaccard = sum(r.get('jaccard', 0) for r in successful_results) / len(successful_results)
        
        print("-"*80)
        print(f"{'平均值':<15} {'':<20} "
              f"{avg_precision:<12.4f} "
              f"{avg_recall:<12.4f} "
              f"{avg_f1:<12.4f} "
              f"{avg_jaccard:<12.4f}")
    
    if failed_results:
        print("\n" + "-"*80)
        print("失败评估的项目:")
        print("-"*80)
        for result in failed_results:
            print(f"  {result.get('project_id', '')} ({result.get('project_name', '')}): {result.get('error_message', '')}")
    
    print("="*80 + "\n")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python batch_evaluate.py <config_file>")
        print("示例: python batch_evaluate.py config.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # 加载配置
    config = load_config(config_file)
    
    # 获取评估设置
    eval_settings = config.get('evaluation_settings', {})
    max_length = eval_settings.get('max_length', 5)
    use_sampling = eval_settings.get('use_sampling', False)
    sample_size = eval_settings.get('sample_size', 10000)
    random_seed = eval_settings.get('random_seed', 42)
    
    # 创建评估器
    evaluator = SequenceEvaluator(
        max_length=max_length,
        use_sampling=use_sampling,
        sample_size=sample_size,
        random_seed=random_seed
    )
    
    # 获取项目列表（只评估enabled的项目）
    projects = config.get('projects', [])
    enabled_projects = [p for p in projects if p.get('enabled', False)]
    
    if not enabled_projects:
        logger.warning("配置文件中没有启用的项目")
        sys.exit(1)
    
    logger.info(f"找到 {len(enabled_projects)} 个启用的项目")
    
    # 批量评估
    results = []
    for project in enabled_projects:
        result = evaluate_project(project, evaluator)
        results.append(result)
    
    # 保存结果
    output_settings = config.get('output_settings', {})
    output_dir = output_settings.get('output_dir', 'results')
    
    if output_settings.get('save_json', True):
        json_filename = output_settings.get('json_filename', 'evaluation_results.json')
        json_path = os.path.join(output_dir, json_filename)
        save_results_json(results, json_path)
    
    if output_settings.get('save_csv', True):
        csv_filename = output_settings.get('csv_filename', 'evaluation_results.csv')
        csv_path = os.path.join(output_dir, csv_filename)
        save_results_csv(results, csv_path)
    
    # 打印汇总
    print_summary(results)
    
    logger.info("批量评估完成")


if __name__ == '__main__':
    main()
