# 批量评估使用说明

## 功能概述

批量评估工具支持通过配置文件批量评估多个项目，并提供丰富的量化指标和格式化输出。

## 主要特性

1. **配置文件驱动**: 通过JSON配置文件管理所有项目
2. **项目选择**: 通过`enabled`字段控制评估哪些项目
3. **丰富的量化指标**: 包括Precision、Recall、F1、Jaccard相似度、序列长度分布等
4. **多种输出格式**: 支持JSON和CSV格式的结果输出
5. **批量处理**: 一次性评估多个项目并生成汇总报告

## 配置文件格式

配置文件`config.json`的格式如下：

```json
{
  "evaluation_settings": {
    "max_length": 5,           // 最大序列长度k
    "use_sampling": false,     // 是否使用采样模式
    "sample_size": 10000,      // 采样大小（如果使用采样）
    "random_seed": 42          // 随机种子
  },
  "projects": [
    {
      "id": "project_1",       // 项目ID（唯一标识）
      "name": "项目1",         // 项目名称
      "enabled": true,         // 是否启用（true=评估，false=跳过）
      "ground_truth_file": "tests/data/ground_truth.txt",
      "prediction_file": "tests/data/prediction.txt"
    },
    // ... 更多项目
  ],
  "output_settings": {
    "output_dir": "results",                    // 结果输出目录
    "save_json": true,                          // 是否保存JSON格式
    "save_csv": true,                           // 是否保存CSV格式
    "json_filename": "evaluation_results.json", // JSON文件名
    "csv_filename": "evaluation_results.csv"    // CSV文件名
  }
}
```

## 使用方法

### 1. 准备配置文件

编辑`config.json`文件，配置你的8个项目：

```json
{
  "evaluation_settings": {
    "max_length": 5,
    "use_sampling": false
  },
  "projects": [
    {
      "id": "project_1",
      "name": "项目1",
      "enabled": true,
      "ground_truth_file": "path/to/project1/ground_truth.txt",
      "prediction_file": "path/to/project1/prediction.txt"
    },
    {
      "id": "project_2",
      "name": "项目2",
      "enabled": true,
      "ground_truth_file": "path/to/project2/ground_truth.txt",
      "prediction_file": "path/to/project2/prediction.txt"
    }
    // ... 添加更多项目，最多8个
  ],
  "output_settings": {
    "output_dir": "results",
    "save_json": true,
    "save_csv": true
  }
}
```

### 2. 运行批量评估

```bash
python batch_evaluate.py config.json
```

### 3. 查看结果

评估完成后，结果会保存在`results/`目录下：

- **evaluation_results.json**: 详细的JSON格式结果，包含所有评估指标
- **evaluation_results.csv**: CSV格式结果，便于在Excel等工具中分析

控制台会输出评估汇总报告，包括：
- 总项目数和成功/失败数量
- 每个项目的详细指标
- 平均值统计

## 评估指标说明

### 核心指标

- **Precision**: 预测模型认为合法的序列中，有多少在ground truth中也允许
  - 公式: `|L_pred(k) ∩ L_gt(k)| / |L_pred(k)|`
  
- **Recall**: ground truth允许的序列中，有多少被预测模型也允许
  - 公式: `|L_pred(k) ∩ L_gt(k)| / |L_gt(k)|`
  
- **F1分数**: Precision和Recall的调和平均
  - 公式: `2 * Precision * Recall / (Precision + Recall)`
  
- **Jaccard相似度**: 两个序列集合的交集与并集的比值
  - 公式: `|L_pred(k) ∩ L_gt(k)| / |L_pred(k) ∪ L_gt(k)|`

### 序列统计

- **num_sequences_gt**: Ground truth序列总数
- **num_sequences_pred**: Prediction序列总数
- **num_intersection**: 交集序列数
- **num_only_in_pred**: 只在Pred中出现的序列数
- **num_only_in_gt**: 只在GT中出现的序列数
- **num_union**: 并集序列数

### 序列长度统计

- **min_length**: 最短序列长度
- **max_length**: 最长序列长度
- **avg_length**: 平均序列长度
- **length_distribution**: 各长度的序列数量分布

### 图统计信息

- **gt_num_nodes/pred_num_nodes**: 节点数
- **gt_num_edges/pred_num_edges**: 边数

### 覆盖率

- **coverage_gt**: GT序列的覆盖率
- **coverage_pred**: Pred序列的覆盖率

## 输出文件格式

### JSON格式

JSON文件包含完整的评估结果，结构如下：

```json
{
  "evaluation_time": "2026-01-12T15:27:58.591044",
  "total_projects": 2,
  "successful_projects": 2,
  "failed_projects": 0,
  "results": [
    {
      "project_id": "project_1",
      "project_name": "项目1",
      "status": "success",
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "jaccard": 1.0,
      // ... 更多指标
    }
  ]
}
```

### CSV格式

CSV文件包含所有项目的关键指标，可以直接在Excel中打开分析。包含以下列：

- project_id, project_name, status
- precision, recall, f1, jaccard
- num_sequences_gt, num_sequences_pred, num_intersection
- num_only_in_pred, num_only_in_gt
- gt_num_nodes, gt_num_edges, pred_num_nodes, pred_num_edges
- coverage_gt, coverage_pred
- gt_avg_seq_length, pred_avg_seq_length
- gt_min_seq_length, gt_max_seq_length
- pred_min_seq_length, pred_max_seq_length
- max_length, use_sampling
- evaluation_time, error_message

## 项目选择

通过在配置文件中设置`enabled`字段来控制评估哪些项目：

- `"enabled": true` - 评估该项目
- `"enabled": false` - 跳过该项目

例如，如果你只想评估项目1、2、3：

```json
{
  "projects": [
    {"id": "project_1", "enabled": true, ...},
    {"id": "project_2", "enabled": true, ...},
    {"id": "project_3", "enabled": true, ...},
    {"id": "project_4", "enabled": false, ...},
    // ... 其他项目设为false
  ]
}
```

## 错误处理

如果某个项目的文件不存在或评估过程中出错：

- 该项目的`status`会被设置为`"error"`
- `error_message`字段会包含错误信息
- 其他项目的评估会继续进行
- 错误信息会在控制台和结果文件中记录

## 注意事项

1. **文件路径**: 配置文件中的路径可以是相对路径或绝对路径
2. **文件编码**: 输入文件应为UTF-8编码
3. **性能考虑**: 对于大型图，建议使用采样模式（`use_sampling: true`）
4. **结果目录**: 结果目录会自动创建，如果不存在的话

## 示例

### 示例1: 评估所有启用的项目

```bash
python batch_evaluate.py config.json
```

### 示例2: 使用采样模式评估大型图

在配置文件中设置：

```json
{
  "evaluation_settings": {
    "max_length": 6,
    "use_sampling": true,
    "sample_size": 20000
  }
}
```

## 常见问题

**Q: 如何只评估部分项目？**
A: 在配置文件中将不需要评估的项目设置为`"enabled": false`

**Q: 如何更改输出目录？**
A: 在配置文件的`output_settings`中修改`output_dir`字段

**Q: 评估结果在哪里？**
A: 结果保存在`results/`目录下（或配置文件中指定的目录）

**Q: 可以同时保存JSON和CSV吗？**
A: 可以，在`output_settings`中同时设置`save_json: true`和`save_csv: true`
