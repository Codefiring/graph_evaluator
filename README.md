# 图编辑距离评估算法

这是一个用于评估预测图与ground truth图之间相似度的Python工具。算法能够忽略节点名称的差异，专注于图的结构和边标签的匹配。

## 功能特点

- ✅ 支持忽略节点名称差异（只关注图结构）
- ✅ 基于图编辑距离的评估方法
- ✅ **有界序列评估方法**（新增）：比较可接受的ioctl序列集合
- ✅ 详细的日志输出，便于监控算法运行
- ✅ 支持有向多重图（带标签的边）
- ✅ 自动节点映射算法

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
graph_evaluator/
├── graph_evaluator/          # 源代码包
│   ├── __init__.py
│   ├── evaluator.py          # 图编辑距离评估算法
│   └── sequence_evaluator.py # 有界序列评估算法（新增）
├── tests/                    # 测试目录
│   ├── __init__.py
│   ├── test_evaluator.py     # 基础测试
│   ├── test_complex.py       # 复杂用例测试
│   ├── test_node_ignoring.py # 节点名称忽略测试
│   ├── test_sequence_evaluator.py # 序列评估测试（新增）
│   └── data/                 # 测试数据
│       ├── ground_truth.txt
│       ├── prediction.txt
│       └── ...
├── logs/                     # 日志目录
├── main.py                   # 图编辑距离评估主入口文件
├── main_sequence.py          # 序列评估主入口文件（新增）
├── README.md
└── requirements.txt
```

## 使用方法

### 1. 准备数据文件

创建两个txt格式的文件，每行一个三元组：

**ground_truth.txt** (标准答案):
```
"S1", "operator1", "S2"
"S2", "operator1", "S2"
"S1", "operator1", "S5"
"S5", "operator2", "S2"
"S2", "operator4", "S3"
```

**prediction.txt** (预测结果):
```
"tst", "operator1", "ggg"
"ggg", "operator1", "ggg"
"tst", "operator1", "S5"
"S5", "operator2", "ggg"
"ggg", "operator4", "S3"
```

### 2. 运行评估

#### 方式一：图编辑距离评估

使用主入口文件：

```bash
python main.py <ground_truth_file> <prediction_file>
```

示例：
```bash
python main.py tests/data/ground_truth.txt tests/data/prediction.txt
```

#### 方式二：有界序列评估（新增）

使用序列评估主入口文件：

```bash
python main_sequence.py <ground_truth_file> <prediction_file> [max_length] [--sampling]
```

参数说明：
- `max_length`: 最大序列长度k（可选，默认5）
- `--sampling`: 使用采样模式（可选，用于分支过多的情况）

示例：
```bash
# 使用默认最大长度5
python main_sequence.py tests/data/ground_truth.txt tests/data/prediction.txt

# 指定最大长度为6
python main_sequence.py tests/data/ground_truth.txt tests/data/prediction.txt 6

# 使用采样模式
python main_sequence.py tests/data/ground_truth.txt tests/data/prediction.txt 5 --sampling
```

#### 方式三：在代码中使用

**图编辑距离评估：**

```python
from graph_evaluator import GraphEvaluator

evaluator = GraphEvaluator()
result = evaluator.evaluate('ground_truth.txt', 'prediction.txt')
print(f"边相似度: {result['edge_similarity']:.4f}")
```

**有界序列评估：**

```python
from graph_evaluator import SequenceEvaluator

evaluator = SequenceEvaluator(max_length=5, use_sampling=False)
result = evaluator.evaluate('ground_truth.txt', 'prediction.txt')
print(f"序列级 Precision: {result['precision']:.4f}")
print(f"序列级 Recall: {result['recall']:.4f}")
print(f"F1分数: {result['f1']:.4f}")
```

### 3. 运行测试

```bash
# 运行基础测试
python tests/test_evaluator.py

# 运行复杂用例测试
python tests/test_complex.py

# 运行节点名称忽略测试
python tests/test_node_ignoring.py

# 运行序列评估测试（新增）
python tests/test_sequence_evaluator.py
```

### 4. 查看结果

- 控制台会输出评估结果摘要
- 图编辑距离评估的详细日志保存在 `logs/graph_evaluation.log` 文件中
- 序列评估的详细日志保存在 `logs/sequence_evaluation.log` 文件中

## 评估指标

### 图编辑距离评估指标

- **总编辑距离**: 将预测图转换为ground truth图所需的总成本
- **节点编辑成本**: 添加/删除节点的成本
- **边编辑成本**: 添加/删除边的成本
- **边相似度**: 共同边数 / 总边数的比例（Jaccard相似度）
- **节点映射**: 预测图中的节点到ground truth图中节点的映射关系

### 有界序列评估指标（新增）

- **序列级 Precision**: `|L_pred(k) ∩ L_gt(k)| / |L_pred(k)|` - 预测模型认为合法的序列中，有多少在ground truth中也允许
- **序列级 Recall**: `|L_pred(k) ∩ L_gt(k)| / |L_gt(k)|` - ground truth允许的序列中，有多少被预测模型也允许
- **F1分数**: Precision和Recall的调和平均
- **序列数量统计**: Ground truth序列数、Prediction序列数、交集序列数

## 算法原理

### 图编辑距离算法

1. **节点映射**: 基于节点的结构特征（入度、出度、边标签分布）找到最佳节点匹配
2. **图标准化**: 将两个图映射到相同的节点空间
3. **编辑距离计算**: 比较标准化后的图，计算需要添加/删除的边和节点

### 有界序列评估算法（新增）

有时候两张图的拓扑有些不同，但可接受的ioctl序列集合很接近，这在fuzzing场景中更重要。

算法流程：

1. **固定最大长度k**: 只看长度≤k的调用序列（默认k=5）
2. **生成序列集合**:
   - 从初始状态（入度为0的节点）出发
   - 遍历所有可能的ioctl序列（到长度k为止）
   - 对ground truth得到集合 L_gt(k)
   - 对预测状态机得到集合 L_pred(k)
3. **节点映射**: 将预测图的节点映射到ground truth图的节点空间（确保语义一致性）
4. **计算指标**:
   - Precision = |L_pred(k) ∩ L_gt(k)| / |L_pred(k)|
   - Recall = |L_pred(k) ∩ L_gt(k)| / |L_gt(k)|
5. **采样优化**: 如果分支超级多，可以启用采样模式，通过随机路径近似估计这些值，而不是全部穷举

## 示例输出

### 图编辑距离评估输出

```
评估结果:
  总编辑距离: 0.0
  节点编辑成本: 0.0
  边编辑成本: 0.0
  需要添加的节点数: 0
  需要删除的节点数: 0
  需要添加的边数: 0
  需要删除的边数: 0
  共同边数: 5
  边相似度: 1.0000
```

### 有界序列评估输出（新增）

```
评估结果汇总:
============================================================
序列级 Precision: 1.0000
序列级 Recall: 1.0000
F1分数: 1.0000

Ground Truth序列数: 10
Prediction序列数: 10
交集序列数: 10
最大序列长度: 5
使用采样: False
============================================================
```

## 自定义参数

### 图编辑距离评估参数

```python
evaluator = GraphEvaluator(
    node_cost=1.0,  # 节点插入/删除的成本
    edge_cost=1.0   # 边插入/删除的成本
)
```

### 有界序列评估参数（新增）

```python
evaluator = SequenceEvaluator(
    max_length=5,        # 最大序列长度k（默认5）
    use_sampling=False,  # 是否使用采样模式（默认False）
    sample_size=10000,   # 采样大小（如果使用采样）
    random_seed=42       # 随机种子（用于可重复性）
)
```

## 注意事项

- 文件编码应为UTF-8
- 三元组格式：`"source", "edge_label", "target"`
- 支持多重边（相同节点对可以有多个不同标签的边）
- 算法会自动处理节点名称的差异
- 日志文件会自动保存到 `logs/` 目录

