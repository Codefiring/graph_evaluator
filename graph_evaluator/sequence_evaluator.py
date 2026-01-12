"""
有界序列评估算法
用于评估预测状态机与ground truth状态机之间可接受的ioctl序列集合的相似度
通过比较长度≤k的所有可能的调用序列来计算precision和recall
"""

import logging
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, deque
import random
import os

# 配置日志
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'sequence_evaluation.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SequenceEvaluator:
    """有界序列评估器"""
    
    def __init__(self, max_length: int = 5, use_sampling: bool = False, sample_size: int = 10000, random_seed: int = 42):
        """
        初始化序列评估器
        
        Args:
            max_length: 最大序列长度k（默认5）
            use_sampling: 是否使用采样（当分支过多时）
            sample_size: 采样大小（如果使用采样）
            random_seed: 随机种子
        """
        self.max_length = max_length
        self.use_sampling = use_sampling
        self.sample_size = sample_size
        self.random_seed = random_seed
        random.seed(random_seed)
        logger.info(f"初始化SequenceEvaluator: max_length={max_length}, use_sampling={use_sampling}, sample_size={sample_size}")
    
    def parse_triples(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        解析三元组文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            三元组列表 [(source, edge_label, target), ...]
        """
        logger.info(f"开始解析文件: {file_path}")
        triples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析三元组，支持引号格式
                    parts = [p.strip().strip('"\'') for p in line.split(',')]
                    if len(parts) >= 3:
                        source = parts[0].strip()
                        edge_label = parts[1].strip()
                        target = parts[2].strip()
                        triples.append((source, edge_label, target))
                    else:
                        logger.warning(f"文件 {file_path} 第 {line_num} 行格式不正确: {line}")
            
            logger.info(f"成功解析 {len(triples)} 个三元组")
            return triples
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时出错: {e}")
            raise
    
    def build_graph(self, triples: List[Tuple[str, str, str]]) -> nx.MultiDiGraph:
        """
        从三元组构建有向多重图
        
        Args:
            triples: 三元组列表
            
        Returns:
            NetworkX有向多重图
        """
        logger.info("开始构建图结构")
        G = nx.MultiDiGraph()
        
        for source, edge_label, target in triples:
            G.add_edge(source, target, label=edge_label)
        
        logger.info(f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        return G
    
    def find_initial_nodes(self, graph: nx.MultiDiGraph) -> List[str]:
        """
        找到初始节点（入度为0的节点，如果没有则返回所有节点）
        
        Args:
            graph: 图对象
            
        Returns:
            初始节点列表
        """
        nodes = list(graph.nodes())
        if not nodes:
            return []
        
        # 找到入度为0的节点
        initial_nodes = [node for node in nodes if graph.in_degree(node) == 0]
        
        # 如果没有入度为0的节点，则从所有节点开始
        if not initial_nodes:
            logger.warning("没有找到入度为0的节点，将从所有节点开始生成序列")
            initial_nodes = nodes
        else:
            logger.info(f"找到 {len(initial_nodes)} 个初始节点: {initial_nodes}")
        
        return initial_nodes
    
    def generate_sequences_exhaustive(self, graph: nx.MultiDiGraph) -> Set[Tuple[str, ...]]:
        """
        穷举生成所有长度≤k的序列
        
        Args:
            graph: 图对象
            
        Returns:
            所有可能的序列集合（每个序列是一个元组）
        """
        logger.info(f"开始穷举生成长度≤{self.max_length}的所有序列")
        sequences = set()
        initial_nodes = self.find_initial_nodes(graph)
        
        # 使用BFS生成所有序列
        queue = deque()
        
        # 从每个初始节点开始
        for init_node in initial_nodes:
            queue.append((init_node, ()))  # (当前节点, 当前序列)
            sequences.add(())  # 空序列也计入
        
        while queue:
            current_node, current_seq = queue.popleft()
            
            # 如果序列长度已达到最大长度，停止扩展
            if len(current_seq) >= self.max_length:
                continue
            
            # 获取当前节点的所有出边
            out_edges = list(graph.out_edges(current_node, data=True))
            
            for _, next_node, data in out_edges:
                edge_label = data.get('label', '')
                new_seq = current_seq + (edge_label,)
                
                # 如果新序列未出现过，加入集合并继续扩展
                if new_seq not in sequences:
                    sequences.add(new_seq)
                    queue.append((next_node, new_seq))
        
        logger.info(f"穷举生成完成: 共 {len(sequences)} 个序列")
        return sequences
    
    def generate_sequences_sampling(self, graph: nx.MultiDiGraph) -> Set[Tuple[str, ...]]:
        """
        通过采样生成序列（用于分支过多的情况）
        
        Args:
            graph: 图对象
            
        Returns:
            采样得到的序列集合
        """
        logger.info(f"开始采样生成序列 (采样大小: {self.sample_size})")
        sequences = set()
        initial_nodes = self.find_initial_nodes(graph)
        
        sequences.add(())  # 空序列
        
        for _ in range(self.sample_size):
            # 随机选择一个初始节点
            current_node = random.choice(initial_nodes)
            current_seq = ()
            
            # 随机游走生成序列
            for _ in range(self.max_length):
                out_edges = list(graph.out_edges(current_node, data=True))
                
                if not out_edges:
                    break
                
                # 随机选择一条边
                _, next_node, data = random.choice(out_edges)
                edge_label = data.get('label', '')
                current_seq = current_seq + (edge_label,)
                sequences.add(current_seq)
                
                current_node = next_node
        
        logger.info(f"采样生成完成: 共 {len(sequences)} 个序列")
        return sequences
    
    def generate_sequences(self, graph: nx.MultiDiGraph) -> Set[Tuple[str, ...]]:
        """
        生成所有长度≤k的序列（根据配置选择穷举或采样）
        
        Args:
            graph: 图对象
            
        Returns:
            序列集合
        """
        # 估算序列数量（简单启发式：如果出边平均数量>2且节点数>10，可能分支过多）
        if self.use_sampling:
            return self.generate_sequences_sampling(graph)
        
        # 先尝试穷举，如果发现可能分支过多，切换到采样
        total_out_edges = sum(graph.out_degree(node) for node in graph.nodes())
        avg_out_degree = total_out_edges / len(graph.nodes()) if graph.nodes() else 0
        
        # 简单估计：如果平均出度较高，可能组合爆炸
        estimated_sequences = (avg_out_degree ** (self.max_length + 1)) if avg_out_degree > 0 else 0
        
        if estimated_sequences > 100000:  # 如果估计序列数过多，使用采样
            logger.warning(f"估计序列数过多 ({estimated_sequences:.0f})，切换到采样模式")
            return self.generate_sequences_sampling(graph)
        else:
            return self.generate_sequences_exhaustive(graph)
    
    def get_node_signature(self, graph: nx.MultiDiGraph, node: str) -> Dict:
        """
        获取节点的结构特征签名（用于节点匹配，复用GraphEvaluator的逻辑）
        """
        in_edges = list(graph.in_edges(node, data=True))
        out_edges = list(graph.out_edges(node, data=True))
        
        in_labels = defaultdict(int)
        out_labels = defaultdict(int)
        
        for _, _, data in in_edges:
            label = data.get('label', '')
            in_labels[label] += 1
        
        for _, _, data in out_edges:
            label = data.get('label', '')
            out_labels[label] += 1
        
        return {
            'in_degree': len(in_edges),
            'out_degree': len(out_edges),
            'in_labels': dict(in_labels),
            'out_labels': dict(out_labels),
            'total_degree': len(in_edges) + len(out_edges)
        }
    
    def find_node_mapping(self, G_gt: nx.MultiDiGraph, G_pred: nx.MultiDiGraph) -> Dict[str, str]:
        """
        找到两个图之间的最佳节点映射（复用GraphEvaluator的逻辑）
        
        Args:
            G_gt: ground truth图
            G_pred: 预测结果图
            
        Returns:
            从G_pred节点到G_gt节点的映射字典
        """
        logger.info("开始寻找节点映射")
        
        nodes_gt = list(G_gt.nodes())
        nodes_pred = list(G_pred.nodes())
        
        # 构建边集合（边标签 -> (源节点, 目标节点)的映射）
        edges_by_label_gt = defaultdict(list)
        edges_by_label_pred = defaultdict(list)
        
        for u, v, data in G_gt.edges(data=True):
            label = data.get('label', '')
            edges_by_label_gt[label].append((u, v))
        
        for u, v, data in G_pred.edges(data=True):
            label = data.get('label', '')
            edges_by_label_pred[label].append((u, v))
        
        # 初始化映射
        mapping = {}  # G_pred节点 -> G_gt节点
        used_nodes_gt = set()
        
        # 通过匹配的边来推断节点映射
        for label in set(edges_by_label_gt.keys()) & set(edges_by_label_pred.keys()):
            edges_gt = edges_by_label_gt[label]
            edges_pred = edges_by_label_pred[label]
            
            for u_pred, v_pred in edges_pred:
                if u_pred in mapping and v_pred in mapping:
                    continue
                
                best_match = None
                best_score = -1
                
                for u_gt, v_gt in edges_gt:
                    if u_gt in used_nodes_gt or v_gt in used_nodes_gt:
                        continue
                    
                    score = 0
                    
                    if u_pred in mapping:
                        if mapping[u_pred] == u_gt:
                            score += 10
                        else:
                            continue
                    if v_pred in mapping:
                        if mapping[v_pred] == v_gt:
                            score += 10
                        else:
                            continue
                    
                    sig_u_gt = self.get_node_signature(G_gt, u_gt)
                    sig_u_pred = self.get_node_signature(G_pred, u_pred)
                    sig_v_gt = self.get_node_signature(G_gt, v_gt)
                    sig_v_pred = self.get_node_signature(G_pred, v_pred)
                    
                    if sig_u_gt['in_degree'] == sig_u_pred['in_degree']:
                        score += 1
                    if sig_u_gt['out_degree'] == sig_u_pred['out_degree']:
                        score += 1
                    if sig_u_gt['in_labels'] == sig_u_pred['in_labels']:
                        score += 2
                    if sig_u_gt['out_labels'] == sig_u_pred['out_labels']:
                        score += 2
                    
                    if sig_v_gt['in_degree'] == sig_v_pred['in_degree']:
                        score += 1
                    if sig_v_gt['out_degree'] == sig_v_pred['out_degree']:
                        score += 1
                    if sig_v_gt['in_labels'] == sig_v_pred['in_labels']:
                        score += 2
                    if sig_v_gt['out_labels'] == sig_v_pred['out_labels']:
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_match = (u_gt, v_gt)
                
                if best_match and best_score > 0:
                    u_gt, v_gt = best_match
                    if u_pred not in mapping and u_gt not in used_nodes_gt:
                        mapping[u_pred] = u_gt
                        used_nodes_gt.add(u_gt)
                    if v_pred not in mapping and v_gt not in used_nodes_gt:
                        mapping[v_pred] = v_gt
                        used_nodes_gt.add(v_gt)
        
        # 处理未映射的节点
        unmatched_pred = [n for n in nodes_pred if n not in mapping]
        unmatched_gt = [n for n in nodes_gt if n not in used_nodes_gt]
        
        signatures_gt = {node: self.get_node_signature(G_gt, node) for node in unmatched_gt}
        signatures_pred = {node: self.get_node_signature(G_pred, node) for node in unmatched_pred}
        
        for node_pred in unmatched_pred:
            sig_pred = signatures_pred[node_pred]
            best_match = None
            best_score = -1
            
            for node_gt in unmatched_gt:
                if node_gt in used_nodes_gt:
                    continue
                
                sig_gt = signatures_gt[node_gt]
                score = 0
                if sig_gt['in_degree'] == sig_pred['in_degree']:
                    score += 2
                if sig_gt['out_degree'] == sig_pred['out_degree']:
                    score += 2
                if sig_gt['in_labels'] == sig_pred['in_labels']:
                    score += 3
                if sig_gt['out_labels'] == sig_pred['out_labels']:
                    score += 3
                
                if score > best_score:
                    best_score = score
                    best_match = node_gt
            
            if best_match and best_score > 0:
                mapping[node_pred] = best_match
                used_nodes_gt.add(best_match)
        
        # 处理剩余节点
        remaining_pred = [n for n in nodes_pred if n not in mapping]
        remaining_gt = [n for n in nodes_gt if n not in used_nodes_gt]
        
        for i, node_pred in enumerate(remaining_pred):
            if i < len(remaining_gt):
                mapping[node_pred] = remaining_gt[i]
        
        logger.info(f"节点映射完成: {len(mapping)} 个节点被映射")
        return mapping
    
    def normalize_graph(self, graph: nx.MultiDiGraph, node_mapping: Dict[str, str]) -> nx.MultiDiGraph:
        """
        使用节点映射标准化图（将pred图的节点映射到gt图的节点）
        
        Args:
            graph: 原始图
            node_mapping: 节点映射字典
            
        Returns:
            标准化后的图
        """
        normalized = nx.MultiDiGraph()
        
        for u, v, data in graph.edges(data=True):
            new_u = node_mapping.get(u, u)
            new_v = node_mapping.get(v, v)
            normalized.add_edge(new_u, new_v, label=data.get('label', ''))
        
        return normalized
    
    def evaluate(self, ground_truth_file: str, prediction_file: str) -> Dict:
        """
        评估预测结果
        
        Args:
            ground_truth_file: ground truth文件路径
            prediction_file: 预测结果文件路径
            
        Returns:
            评估结果字典，包含precision和recall
        """
        logger.info("=" * 60)
        logger.info("开始有界序列评估流程")
        logger.info(f"Ground Truth文件: {ground_truth_file}")
        logger.info(f"预测结果文件: {prediction_file}")
        logger.info(f"最大序列长度: {self.max_length}")
        logger.info("=" * 60)
        
        # 解析文件
        triples_gt = self.parse_triples(ground_truth_file)
        triples_pred = self.parse_triples(prediction_file)
        
        # 构建图
        G_gt = self.build_graph(triples_gt)
        G_pred = self.build_graph(triples_pred)
        
        # 找到节点映射
        node_mapping = self.find_node_mapping(G_gt, G_pred)
        
        # 标准化pred图（将节点映射到gt图的节点空间）
        G_pred_normalized = self.normalize_graph(G_pred, node_mapping)
        
        # 生成序列
        logger.info("生成ground truth序列...")
        sequences_gt = self.generate_sequences(G_gt)
        logger.info(f"Ground truth序列数: {len(sequences_gt)}")
        
        logger.info("生成prediction序列...")
        sequences_pred = self.generate_sequences(G_pred_normalized)
        logger.info(f"Prediction序列数: {len(sequences_pred)}")
        
        # 计算交集
        intersection = sequences_pred & sequences_gt
        only_in_pred = sequences_pred - sequences_gt  # 只在pred中出现的序列
        only_in_gt = sequences_gt - sequences_pred    # 只在gt中出现的序列
        
        # 计算precision和recall
        if len(sequences_pred) > 0:
            precision = len(intersection) / len(sequences_pred)
        else:
            precision = 0.0
        
        if len(sequences_gt) > 0:
            recall = len(intersection) / len(sequences_gt)
        else:
            recall = 0.0
        
        # 计算F1分数
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # 计算Jaccard相似度
        union = sequences_pred | sequences_gt
        if len(union) > 0:
            jaccard = len(intersection) / len(union)
        else:
            jaccard = 1.0
        
        # 计算序列长度统计
        def get_sequence_stats(seq_set):
            if not seq_set:
                return {
                    'min_length': 0,
                    'max_length': 0,
                    'avg_length': 0.0,
                    'length_distribution': {}
                }
            lengths = [len(seq) for seq in seq_set]
            length_dist = {}
            for length in lengths:
                length_dist[length] = length_dist.get(length, 0) + 1
            return {
                'min_length': min(lengths),
                'max_length': max(lengths),
                'avg_length': sum(lengths) / len(lengths),
                'length_distribution': length_dist
            }
        
        gt_seq_stats = get_sequence_stats(sequences_gt)
        pred_seq_stats = get_sequence_stats(sequences_pred)
        intersection_seq_stats = get_sequence_stats(intersection)
        
        # 图统计信息
        gt_num_nodes = G_gt.number_of_nodes()
        gt_num_edges = G_gt.number_of_edges()
        pred_num_nodes = G_pred.number_of_nodes()
        pred_num_edges = G_pred.number_of_edges()
        
        # 计算覆盖率
        coverage_gt = len(intersection) / len(sequences_gt) if len(sequences_gt) > 0 else 0.0
        coverage_pred = len(intersection) / len(sequences_pred) if len(sequences_pred) > 0 else 0.0
        
        result = {
            # 核心指标
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'jaccard': jaccard,
            
            # 序列数量
            'num_sequences_gt': len(sequences_gt),
            'num_sequences_pred': len(sequences_pred),
            'num_intersection': len(intersection),
            'num_only_in_pred': len(only_in_pred),
            'num_only_in_gt': len(only_in_gt),
            'num_union': len(union),
            
            # 序列长度统计
            'gt_sequence_stats': gt_seq_stats,
            'pred_sequence_stats': pred_seq_stats,
            'intersection_sequence_stats': intersection_seq_stats,
            
            # 图统计信息
            'gt_num_nodes': gt_num_nodes,
            'gt_num_edges': gt_num_edges,
            'pred_num_nodes': pred_num_nodes,
            'pred_num_edges': pred_num_edges,
            
            # 覆盖率
            'coverage_gt': coverage_gt,
            'coverage_pred': coverage_pred,
            
            # 配置信息
            'max_length': self.max_length,
            'use_sampling': self.use_sampling,
            'node_mapping': node_mapping
        }
        
        # 打印结果
        logger.info("=" * 60)
        logger.info("评估结果:")
        logger.info(f"  序列级 Precision: {precision:.4f}")
        logger.info(f"  序列级 Recall: {recall:.4f}")
        logger.info(f"  F1分数: {f1:.4f}")
        logger.info(f"  Jaccard相似度: {jaccard:.4f}")
        logger.info(f"  Ground truth序列数: {len(sequences_gt)}")
        logger.info(f"  Prediction序列数: {len(sequences_pred)}")
        logger.info(f"  交集序列数: {len(intersection)}")
        logger.info(f"  只在Pred中: {len(only_in_pred)}")
        logger.info(f"  只在GT中: {len(only_in_gt)}")
        logger.info(f"  GT图节点数: {gt_num_nodes}, 边数: {gt_num_edges}")
        logger.info(f"  Pred图节点数: {pred_num_nodes}, 边数: {pred_num_edges}")
        logger.info("=" * 60)
        
        return result
