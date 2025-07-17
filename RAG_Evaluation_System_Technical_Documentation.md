# RAG评估系统技术文档

## 概述

本文档详细描述了RAG（检索增强生成）系统的评估框架实现。该系统专门用于评估问答模型的准确率、召回率和F1分数，通过语义相似度匹配实现智能评估，支持WikiQA数据集的自动化测试。

## 系统架构

### 1. 核心组件

#### 功能定位
- **评估类型**：RAG系统端到端评估
- **评估方法**：语义相似度匹配 + 传统指标
- **支持数据**：WikiQA格式数据集
- **设计目标**：自动化、准确的模型性能评估

#### 依赖库
- `sklearn.metrics`：传统机器学习评估指标
- `pandas`：数据处理和分析
- `model_manager`：统一的模型管理接口
- `cosine_similarity`：语义相似度计算

### 2. 评估框架设计

#### 2.1 语义评估系统

**核心功能**
- **文本向量化**：将文本转换为语义向量
- **语义匹配**：基于余弦相似度的语义比较
- **阈值判断**：可配置的匹配阈值
- **批量处理**：支持大规模数据评估

#### 2.2 主评估函数：`evaluate_model()`

**函数签名**
```python
def evaluate_model(qa_chain, tsv_path="WikiQA/WikiQA-test.tsv", max_q=50):
    """
    评估模型准确率、召回率和F1分数
    
    Args:
        qa_chain: RAG问答系统链
        tsv_path: 测试数据集路径
        max_q: 最大评估问题数量
    
    Returns:
        dict: 包含准确率、召回率、F1分数的评估结果
    """
```

**参数说明**

| 参数名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| `qa_chain` | Chain | RAG问答系统实例 | 必需 |
| `tsv_path` | str | WikiQA格式测试文件路径 | "WikiQA/WikiQA-test.tsv" |
| `max_q` | int | 最大评估问题数 | 50 |

**返回值**
```python
{
    'total': int,      # 总问题数
    'correct': int,    # 正确回答数
    'accuracy': float, # 准确率
    'recall': float,   # 召回率
    'f1': float        # F1分数
}
```

### 3. 语义匹配系统

#### 3.1 向量化模块

**函数：`get_embedding()`**
```python
def get_embedding(text, model):
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
        model: 嵌入模型
    
    Returns:
        list: 文本的嵌入向量
    """
```

**技术实现**
- 使用统一的模型管理器获取嵌入模型
- 支持多种嵌入模型后端
- 标准化接口设计

#### 3.2 语义匹配算法

**函数：`is_semantic_match()`**

**算法流程**
```
输入：预测答案 + 真实答案列表 + 嵌入模型 + 阈值
↓
预测答案向量化
↓
遍历所有真实答案
    计算余弦相似度
    如果相似度 >= 阈值
        返回True
↓
返回False（无匹配）
```

**相似度计算**
```python
sim = cosine_similarity([pred_emb], [ans_emb])[0][0]
return sim >= threshold  # 默认阈值0.7
```

### 4. 评估流程详解

#### 4.1 数据加载阶段

**数据格式要求**
- **文件格式**: TSV (Tab-Separated Values)
- **必需字段**: Question, Sentence, Label
- **数据集**: WikiQA标准格式

**错误处理**
```python
try:
    df = pd.read_csv(tsv_path, sep="\t")
except FileNotFoundError:
    print(f"错误：找不到测试文件 {tsv_path}")
    return
except Exception as e:
    print(f"错误：读取测试文件失败 {e}")
    return
```

#### 4.2 数据处理阶段

**问题分组**
```python
grouped = df.groupby("Question")
# 按问题分组，处理每个问题的所有候选答案
```

**进度显示**
```python
print(f"开始评估，预计处理 {min(max_q, len(grouped))} 个问题...")
if (i + 1) % 10 == 0:
    print(f"已处理 {i + 1}/{min(max_q, len(grouped))} 个问题...")
```

#### 4.3 问答测试阶段

**答案提取**
```python
true_answers = group[group["Label"] == 1]["Sentence"].tolist()
if not true_answers:
    continue  # 跳过无正确答案的问题
```

**模型调用**
```python
try:
    prediction = qa_chain.invoke({"query": question})
    pred_text = prediction.get("result", "")
    if not pred_text.strip():
        print(f"警告：问题 [{i + 1}] 获得空回答")
        continue
except Exception as e:
    print(f"问题出错：{question} | {e}")
    continue
```

#### 4.4 结果评估阶段

**匹配判断**
```python
matched = is_semantic_match(pred_text, true_answers, embed_model)
total += 1
correct += int(matched)
```

**指标计算**
```python
accuracy = correct / total
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

### 5. 使用示例

#### 5.1 基础使用

```python
from evaluation import evaluate_model
from qa_system import create_qa_chain

# 创建问答系统
qa_chain = create_qa_chain(vector_db)

# 运行评估
results = evaluate_model(
    qa_chain=qa_chain,
    tsv_path="WikiQA/WikiQA-test.tsv",
    max_q=100
)

# 查看结果
print(f"准确率: {results['accuracy']:.2%}")
print(f"召回率: {results['recall']:.2%}")
print(f"F1分数: {results['f1']:.3f}")
```

#### 5.2 自定义评估

```python
# 调整语义匹配阈值
def custom_evaluate(qa_chain, threshold=0.8):
    from functools import partial
    
    # 使用自定义阈值
    def custom_match(pred, truths, embed_model):
        return is_semantic_match(pred, truths, embed_model, threshold)
    
    # 运行评估...（自定义实现）

# 扩展评估指标
def evaluate_with_additional_metrics(qa_chain):
    results = evaluate_model(qa_chain)
    
    # 添加额外指标
    results['semantic_threshold'] = 0.7
    results['evaluation_method'] = 'cosine_similarity'
    
    return results
```

#### 5.3 批量评估脚本

```python
import json
import time
from evaluation import evaluate_model

def comprehensive_evaluation():
    """综合评估脚本"""
    
    configs = [
        {"max_q": 50, "threshold": 0.7},
        {"max_q": 100, "threshold": 0.75},
        {"max_q": 200, "threshold": 0.8}
    ]
    
    results = []
    
    for config in configs:
        print(f"开始评估配置: {config}")
        start_time = time.time()
        
        # 运行评估
        result = evaluate_model(
            qa_chain=qa_chain,
            max_q=config["max_q"]
        )
        
        if result:
            result.update({
                "config": config,
                "evaluation_time": time.time() - start_time
            })
            results.append(result)
    
    # 保存结果
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results
```

### 6. 性能优化

#### 6.1 计算优化

**批处理向量化**
```python
# 优化建议：批量计算嵌入
def batch_get_embeddings(texts, model):
    """批量获取嵌入向量"""
    return [model.embed_query(text) for text in texts]

# 预计算真实答案嵌入
def precompute_answer_embeddings(true_answers, embed_model):
    """预计算所有真实答案的嵌入"""
    return {ans: get_embedding(ans, embed_model) for ans in true_answers}
```

**缓存机制**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text, model_name):
    """缓存嵌入结果"""
    return model_manager.get_embedding_model().embed_query(text)
```

#### 6.2 内存优化

**流式处理**
```python
def evaluate_streaming(qa_chain, chunk_size=10):
    """流式评估减少内存占用"""
    # 分批处理数据
    pass
```

### 7. 结果分析

#### 7.1 评估报告格式

**标准输出**
```
============================================================
评估结果汇总：
总问题数: 50
正确回答: 35
准确率: 70.00%
召回率: 68.00%
F1 分数: 0.689
============================================================
```

**详细分析**
```python
def analyze_results(results):
    """分析评估结果"""
    analysis = {
        "performance_level": "优秀" if results["accuracy"] > 0.8 else "良好" if results["accuracy"] > 0.6 else "需改进",
        "balance_check": abs(results["precision"] - results["recall"]) < 0.1,
        "semantic_quality": results["f1"] > 0.7,
        "recommendations": []
    }
    
    if results["accuracy"] < 0.6:
        analysis["recommendations"].append("考虑增加训练数据")
        analysis["recommendations"].append("调整语义匹配阈值")
    
    return analysis
```

### 8. 故障排除

#### 8.1 常见问题

**问题1：评估结果异常**
```python
# 诊断代码
if accuracy == 0.0:
    print("检查：所有预测都错误")
    # 可能是语义阈值过高
    
if accuracy == 1.0 and total < 10:
    print("警告：样本量过小，结果可能不可靠")
```

**问题2：性能瓶颈**
```python
# 性能分析
import time

def profile_evaluation():
    start = time.time()
    results = evaluate_model(qa_chain, max_q=10)
    elapsed = time.time() - start
    print(f"评估10个问题耗时: {elapsed:.2f}秒")
    print(f"平均每个问题: {elapsed/10:.2f}秒")
```

#### 8.2 调试工具

**日志增强**
```python
import logging
logging.basicConfig(level=logging.INFO)

def debug_evaluate(qa_chain):
    """带调试信息的评估"""
    def debug_match(pred, truths, embed_model):
        pred_emb = get_embedding(pred, embed_model)
        for ans in truths:
            ans_emb = get_embedding(ans, embed_model)
            sim = cosine_similarity([pred_emb], [ans_emb])[0][0]
            print(f"相似度: {sim:.3f} - {pred[:50]}... vs {ans[:50]}...")
        return is_semantic_match(pred, truths, embed_model)
    
    # 使用调试版本...
```

### 9. 扩展功能

#### 9.1 多维度评估

**增加BLEU分数**
```python
from nltk.translate.bleu_score import sentence_bleu

def evaluate_with_bleu(qa_chain):
    """增加BLEU评估"""
    # 实现BLEU分数计算
    pass
```

**增加ROUGE分数**
```python
from rouge import Rouge

def evaluate_with_rouge(qa_chain):
    """增加ROUGE评估"""
    # 实现ROUGE分数计算
    pass
```

#### 9.2 可视化报告

**生成图表**
```python
import matplotlib.pyplot as plt

def create_evaluation_chart(results):
    """生成评估图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 准确率饼图
    ax1.pie([results['correct'], results['total'] - results['correct']], 
            labels=['正确', '错误'], autopct='%1.1f%%')
    ax1.set_title('回答正确率')
    
    # 指标柱状图
    metrics = ['Accuracy', 'Recall', 'F1']
    values = [results['accuracy'], results['recall'], results['f1']]
    ax2.bar(metrics, values)
    ax2.set_title('评估指标')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('evaluation_chart.png')
```

### 10. 部署建议

#### 10.1 CI/CD集成

**GitHub Actions工作流**
```yaml
# .github/workflows/evaluate.yml
name: RAG Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Evaluation
        run: |
          python -m pip install -r requirements.txt
          python evaluation.py
```

#### 10.2 监控集成

**Prometheus指标**
```python
from prometheus_client import Counter, Histogram

EVALUATION_COUNTER = Counter('rag_evaluations_total', 'Total evaluations')
EVALUATION_ACCURACY = Histogram('rag_accuracy', 'Accuracy distribution')

def monitored_evaluate(qa_chain):
    EVALUATION_COUNTER.inc()
    results = evaluate_model(qa_chain)
    EVALUATION_ACCURACY.observe(results['accuracy'])
    return results
```

## 总结

本RAG评估系统提供了专业、全面的模型性能评估解决方案。通过语义相似度匹配实现智能评估，支持传统指标和语义指标的结合，具备良好的扩展性和实用性。系统设计简洁高效，适用于RAG系统的开发、测试和生产监控等各个阶段，是确保模型质量的重要工具。