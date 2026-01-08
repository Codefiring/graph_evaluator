"""
图编辑距离评估算法
用于评估预测图与ground truth图之间的相似度
支持忽略节点名称差异，只关注图的结构和边标签
"""

import logging
import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GraphEvaluator:
    """图编辑距离评估器"""
    
    def __init__(self, node_cost: float = 1.0, edge_cost: float = 1.0):
        """
        初始化评估器
        
        Args:
            node_cost: 节点插入/删除的成本
            edge_cost: 边插入/删除的成本
        """
        self.node_cost = node_cost
        self.edge_cost = edge_cost
        logger.info(f"初始化GraphEvaluator: node_cost={node_cost}, edge_cost={edge_cost}")
    
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
    
    def get_node_signature(self, graph: nx.MultiDiGraph, node: str) -> Dict:
        """
        获取节点的结构特征签名（用于节点匹配）
        
        Args:
            graph: 图对象
            node: 节点名称
            
        Returns:
            节点特征字典
        """
        in_edges = list(graph.in_edges(node, data=True))
        out_edges = list(graph.out_edges(node, data=True))
        
        # 统计入边和出边的标签分布
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
    
    def find_node_mapping(self, G1: nx.MultiDiGraph, G2: nx.MultiDiGraph) -> Dict[str, str]:
        """
        找到两个图之间的最佳节点映射
        基于边标签匹配和节点结构特征，完全忽略节点名称差异
        
        Args:
            G1: 第一个图（ground truth）
            G2: 第二个图（预测结果）
            
        Returns:
            从G2节点到G1节点的映射字典
        """
        logger.info("开始寻找节点映射（忽略节点名称差异）")
        
        nodes1 = list(G1.nodes())
        nodes2 = list(G2.nodes())
        
        if len(nodes1) != len(nodes2):
            logger.warning(f"节点数量不匹配: G1有{len(nodes1)}个节点, G2有{len(nodes2)}个节点")
        
        # 第一步：基于边的匹配来推断节点映射
        # 构建边集合（边标签 -> (源节点, 目标节点)的映射）
        edges_by_label1 = defaultdict(list)
        edges_by_label2 = defaultdict(list)
        
        for u, v, data in G1.edges(data=True):
            label = data.get('label', '')
            edges_by_label1[label].append((u, v))
        
        for u, v, data in G2.edges(data=True):
            label = data.get('label', '')
            edges_by_label2[label].append((u, v))
        
        # 初始化映射
        mapping = {}  # G2节点 -> G1节点
        reverse_mapping = {}  # G1节点 -> G2节点（用于验证一致性）
        used_nodes1 = set()
        
        # 第二步：通过匹配的边来推断节点映射
        # 对于每个边标签，尝试匹配具有相同标签的边
        for label in set(edges_by_label1.keys()) & set(edges_by_label2.keys()):
            edges1 = edges_by_label1[label]
            edges2 = edges_by_label2[label]
            
            # 对于每条G2的边，找到最匹配的G1的边
            for u2, v2 in edges2:
                if u2 in mapping and v2 in mapping:
                    continue  # 已经映射过了
                
                best_match = None
                best_score = -1
                
                for u1, v1 in edges1:
                    # 检查是否已经被其他节点使用
                    if u1 in used_nodes1 or v1 in used_nodes1:
                        continue
                    
                    # 计算匹配分数
                    score = 0
                    
                    # 如果源节点或目标节点已经映射，检查一致性
                    if u2 in mapping:
                        if mapping[u2] == u1:
                            score += 10  # 强匹配
                        else:
                            continue  # 不一致，跳过
                    if v2 in mapping:
                        if mapping[v2] == v1:
                            score += 10  # 强匹配
                        else:
                            continue  # 不一致，跳过
                    
                    # 基于节点结构特征
                    sig_u1 = self.get_node_signature(G1, u1)
                    sig_u2 = self.get_node_signature(G2, u2)
                    sig_v1 = self.get_node_signature(G1, v1)
                    sig_v2 = self.get_node_signature(G2, v2)
                    
                    # 源节点匹配分数
                    if sig_u1['in_degree'] == sig_u2['in_degree']:
                        score += 1
                    if sig_u1['out_degree'] == sig_u2['out_degree']:
                        score += 1
                    if sig_u1['in_labels'] == sig_u2['in_labels']:
                        score += 2
                    if sig_u1['out_labels'] == sig_u2['out_labels']:
                        score += 2
                    
                    # 目标节点匹配分数
                    if sig_v1['in_degree'] == sig_v2['in_degree']:
                        score += 1
                    if sig_v1['out_degree'] == sig_v2['out_degree']:
                        score += 1
                    if sig_v1['in_labels'] == sig_v2['in_labels']:
                        score += 2
                    if sig_v1['out_labels'] == sig_v2['out_labels']:
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_match = (u1, v1)
                
                if best_match and best_score > 0:
                    u1, v1 = best_match
                    # 添加映射
                    if u2 not in mapping and u1 not in used_nodes1:
                        mapping[u2] = u1
                        reverse_mapping[u1] = u2
                        used_nodes1.add(u1)
                        logger.debug(f"通过边匹配映射节点: {u2} -> {u1} (分数: {best_score})")
                    if v2 not in mapping and v1 not in used_nodes1:
                        mapping[v2] = v1
                        reverse_mapping[v1] = v2
                        used_nodes1.add(v1)
                        logger.debug(f"通过边匹配映射节点: {v2} -> {v1} (分数: {best_score})")
        
        # 第三步：对于未映射的节点，基于结构特征匹配
        unmatched_nodes2 = [n for n in nodes2 if n not in mapping]
        unmatched_nodes1 = [n for n in nodes1 if n not in used_nodes1]
        
        logger.info(f"通过边匹配映射了 {len(mapping)} 个节点，剩余 {len(unmatched_nodes2)} 个节点需要基于结构匹配")
        
        # 计算未匹配节点的特征
        signatures1 = {node: self.get_node_signature(G1, node) for node in unmatched_nodes1}
        signatures2 = {node: self.get_node_signature(G2, node) for node in unmatched_nodes2}
        
        # 基于特征匹配剩余节点
        for node2 in unmatched_nodes2:
            sig2 = signatures2[node2]
            best_match = None
            best_score = -1
            
            for node1 in unmatched_nodes1:
                if node1 in used_nodes1:
                    continue
                
                sig1 = signatures1[node1]
                
                # 计算匹配分数
                score = 0
                if sig1['in_degree'] == sig2['in_degree']:
                    score += 2
                if sig1['out_degree'] == sig2['out_degree']:
                    score += 2
                if sig1['in_labels'] == sig2['in_labels']:
                    score += 3
                if sig1['out_labels'] == sig2['out_labels']:
                    score += 3
                
                if score > best_score:
                    best_score = score
                    best_match = node1
            
            if best_match and best_score > 0:
                mapping[node2] = best_match
                used_nodes1.add(best_match)
                logger.debug(f"通过结构特征映射节点: {node2} -> {best_match} (分数: {best_score})")
        
        # 第四步：处理仍然未匹配的节点（按顺序分配）
        remaining_unmatched2 = [n for n in nodes2 if n not in mapping]
        remaining_unmatched1 = [n for n in nodes1 if n not in used_nodes1]
        
        for i, node2 in enumerate(remaining_unmatched2):
            if i < len(remaining_unmatched1):
                mapping[node2] = remaining_unmatched1[i]
                logger.debug(f"按顺序映射剩余节点: {node2} -> {remaining_unmatched1[i]}")
        
        logger.info(f"节点映射完成: {len(mapping)} 个节点被映射（完全忽略节点名称差异）")
        return mapping
    
    def normalize_graph(self, graph: nx.MultiDiGraph, node_mapping: Dict[str, str] = None) -> nx.MultiDiGraph:
        """
        标准化图（重命名节点）
        
        Args:
            graph: 原始图
            node_mapping: 节点映射字典（如果为None，则使用内部映射）
            
        Returns:
            标准化后的图
        """
        logger.info("开始标准化图")
        normalized = nx.MultiDiGraph()
        
        if node_mapping is None:
            # 如果没有提供映射，使用节点索引作为新名称
            node_list = list(graph.nodes())
            node_mapping = {node: f"N{i}" for i, node in enumerate(node_list)}
        
        # 添加边（使用映射后的节点名称）
        for u, v, data in graph.edges(data=True):
            new_u = node_mapping.get(u, u)
            new_v = node_mapping.get(v, v)
            normalized.add_edge(new_u, new_v, label=data.get('label', ''))
        
        logger.info(f"图标准化完成: {normalized.number_of_nodes()} 个节点, {normalized.number_of_edges()} 条边")
        return normalized
    
    def compute_edit_distance(self, G1: nx.MultiDiGraph, G2: nx.MultiDiGraph) -> Dict:
        """
        计算两个图之间的编辑距离
        
        Args:
            G1: ground truth图
            G2: 预测结果图
            
        Returns:
            包含编辑距离和详细信息的字典
        """
        logger.info("开始计算图编辑距离")
        
        # 找到节点映射
        node_mapping = self.find_node_mapping(G1, G2)
        
        # 标准化G2（使用映射）
        G2_normalized = self.normalize_graph(G2, node_mapping)
        
        # 标准化G1（使用自身节点作为标准）
        G1_normalized = self.normalize_graph(G1, {node: node for node in G1.nodes()})
        
        # 计算边的差异
        edges1 = set()
        for u, v, data in G1_normalized.edges(data=True):
            edge_key = (u, v, data.get('label', ''))
            edges1.add(edge_key)
        
        edges2 = set()
        for u, v, data in G2_normalized.edges(data=True):
            edge_key = (u, v, data.get('label', ''))
            edges2.add(edge_key)
        
        # 计算需要添加和删除的边
        edges_to_add = edges2 - edges1
        edges_to_delete = edges1 - edges2
        edges_common = edges1 & edges2
        
        # 计算节点差异
        nodes1 = set(G1_normalized.nodes())
        nodes2 = set(G2_normalized.nodes())
        nodes_to_add = nodes2 - nodes1
        nodes_to_delete = nodes1 - nodes2
        
        # 计算总编辑距离
        node_edit_cost = len(nodes_to_add) * self.node_cost + len(nodes_to_delete) * self.node_cost
        edge_edit_cost = len(edges_to_add) * self.edge_cost + len(edges_to_delete) * self.edge_cost
        total_distance = node_edit_cost + edge_edit_cost
        
        # 计算相似度指标
        total_edges = len(edges1) + len(edges2)
        if total_edges > 0:
            edge_similarity = 2 * len(edges_common) / total_edges
        else:
            edge_similarity = 1.0 if len(edges1) == 0 and len(edges2) == 0 else 0.0
        
        result = {
            'total_distance': total_distance,
            'node_edit_cost': node_edit_cost,
            'edge_edit_cost': edge_edit_cost,
            'nodes_to_add': len(nodes_to_add),
            'nodes_to_delete': len(nodes_to_delete),
            'edges_to_add': len(edges_to_add),
            'edges_to_delete': len(edges_to_delete),
            'edges_common': len(edges_common),
            'edge_similarity': edge_similarity,
            'node_mapping': node_mapping,
            'edges_to_add_details': list(edges_to_add),
            'edges_to_delete_details': list(edges_to_delete)
        }
        
        logger.info(f"编辑距离计算完成: 总距离={total_distance}, 边相似度={edge_similarity:.4f}")
        return result
    
    def evaluate(self, ground_truth_file: str, prediction_file: str) -> Dict:
        """
        评估预测结果
        
        Args:
            ground_truth_file: ground truth文件路径
            prediction_file: 预测结果文件路径
            
        Returns:
            评估结果字典
        """
        logger.info("=" * 60)
        logger.info("开始图评估流程")
        logger.info(f"Ground Truth文件: {ground_truth_file}")
        logger.info(f"预测结果文件: {prediction_file}")
        logger.info("=" * 60)
        
        # 解析文件
        triples_gt = self.parse_triples(ground_truth_file)
        triples_pred = self.parse_triples(prediction_file)
        
        # 构建图
        G_gt = self.build_graph(triples_gt)
        G_pred = self.build_graph(triples_pred)
        
        # 计算编辑距离
        result = self.compute_edit_distance(G_gt, G_pred)
        
        # 打印详细结果
        logger.info("=" * 60)
        logger.info("评估结果:")
        logger.info(f"  总编辑距离: {result['total_distance']}")
        logger.info(f"  节点编辑成本: {result['node_edit_cost']}")
        logger.info(f"  边编辑成本: {result['edge_edit_cost']}")
        logger.info(f"  需要添加的节点数: {result['nodes_to_add']}")
        logger.info(f"  需要删除的节点数: {result['nodes_to_delete']}")
        logger.info(f"  需要添加的边数: {result['edges_to_add']}")
        logger.info(f"  需要删除的边数: {result['edges_to_delete']}")
        logger.info(f"  共同边数: {result['edges_common']}")
        logger.info(f"  边相似度: {result['edge_similarity']:.4f}")
        logger.info("=" * 60)
        
        if result['edges_to_add'] > 0:
            logger.info(f"需要添加的边: {result['edges_to_add_details']}")
        if result['edges_to_delete'] > 0:
            logger.info(f"需要删除的边: {result['edges_to_delete_details']}")
        
        return result


def main():
    """主函数"""
    evaluator = GraphEvaluator()
    
    # 评估
    result = evaluator.evaluate('ground_truth.txt', 'prediction.txt')
    
    print("\n评估完成！详细日志已保存到 graph_evaluation.log")
    print(f"边相似度: {result['edge_similarity']:.4f}")


if __name__ == '__main__':
    main()

