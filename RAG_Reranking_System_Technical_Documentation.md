# RAG重排序系统技术文档

## 概述

本文档详细描述了RAG（检索增强生成）系统的重排序模块实现。该系统通过多阶段重排序算法优化检索结果，结合关键词匹配、长度适应和对比学习技术，显著提升最终答案的相关性和准确性。

## 系统架构

### 1. 核心组件

#### 1.1 重排序器类型

| 重排序器类型 | 技术方案 | 主要功能 | 适用场景 |
|-------------|----------|----------|----------|
| `SimpleReranker` | 关键词+长度启发式 | 基础重排序 | 通用场景 |
| `ContrastiveReranker` | 对比学习增强 | 负样本抑制 | 训练数据充足场景 |

#### 1.2 功能定位

- **优化目标**: 提升检索结果的相关性和准确性
- **技术路线**: 多维度特征融合 + 机器学习增强
- **设计原则**: 轻量级、可解释、易扩展
- **性能要求**: 低延迟、高准确率

### 2. SimpleReranker 基础重排序器

#### 2.1 初始化配置

```python
SimpleReranker(
    keyword_weight=0.7,   # 关键词匹配权重 (0.0-1.0)
    length_weight=0.3     # 长度适应权重 (0.0-1.0)
)
```

#### 2.2 评分算法

**综合评分公式**
```
FinalScore = keyword_weight × KeywordScore + length_weight × LengthScore
```

**关键词评分算法**
```python
def _calculate_keyword_score(self, query_keywords, doc_text):
    matches = 0
    for keyword in query_keywords:
        if exact_match(keyword, doc_text):
            matches += 1.0      # 精确匹配
        elif partial_match(keyword, doc_text):
            matches += 0.5      # 部分匹配
    return matches / len(query_keywords)
```

**长度评分算法**
```python
def _calculate_length_score(self, query, doc_text):
    # 基于查询长度确定理想文档长度
    query_length = len(query.split())
    if query_length <= 3: ideal = 150
    elif query_length <= 8: ideal = 300
    else: ideal = 500
    
    # 计算长度适应度
    length_diff = abs(len(doc_text) - ideal)
    return max(0, 1 - (length_diff / ideal))
```

#### 2.3 关键词提取

**停用词过滤**
```python
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'what', 'where', 'when', 'why', 'how', 'who', 'which', 'that', 'this'
}
```

**关键词提取流程**
```
输入文本 → 小写转换 → 正则分词 → 停用词过滤 → 短词过滤 → 关键词列表
```

### 3. ContrastiveReranker 对比学习重排序器

#### 3.1 设计原理

**负样本抑制机制**
- **识别负样本**: 基于文本相似度检测
- **分数惩罚**: 负样本分数乘以0.3倍
- **正样本提升**: 非负样本分数乘以1.1倍

#### 3.2 相似度计算

**文本相似度算法**
```python
def _text_similarity(self, text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0
```

**负样本检测**
```python
def _is_similar_to_negatives(self, doc_text, negative_samples):
    doc_clean = normalize_text(doc_text)
    for neg in negative_samples:
        neg_text = normalize_text(neg['text'])
        if similarity(doc_clean, neg_text) > 0.7:
            return True
    return False
```

### 4. 重排序展示系统

#### 4.1 主函数：`show_retrieved_documents_with_rerank()`

**功能流程**
```
检索文档 → 重排序处理 → 类型标注 → 结果展示 → 统计分析
```

**参数说明**
| 参数名 | 类型 | 描述 | 必需 |
|--------|------|------|------|
| `db` | VectorStore | 向量数据库实例 | 是 |
| `question` | str | 查询问题 | 是 |
| `contrastive_data` | dict | 对比学习数据 | 否 |
| `reranker` | Reranker | 重排序器实例 | 否 |

#### 4.2 结果展示格式

**标准输出格式**
```
[重排序文档 1] 🟢 正样本 | 分数: 0.892
   从第3位上升到第1位
   原因: 关键词匹配度高; 长度适中
   内容: Machine learning is a subset of artificial intelligence...
```

**质量分析报告**
```
检索质量分析:
    正样本文档: 4/5 (80.0%)
    负样本文档: 1/5 (20.0%)
    检索质量: 优秀 (80.0分)
    重排序后质量: 92.0分
    重排序改进: +12.0分 (显著提升)
```

### 5. 使用示例

#### 5.1 基础使用

```python
from reranking import SimpleReranker, ContrastiveReranker, create_reranker

# 创建重排序器
simple_reranker = SimpleReranker(keyword_weight=0.8, length_weight=0.2)
contrastive_reranker = ContrastiveReranker()

# 使用工厂函数创建
reranker = create_reranker("simple", keyword_weight=0.7, length_weight=0.3)
```

#### 5.2 完整示例

```python
from reranking import show_retrieved_documents_with_rerank

# 准备数据
question = "What is machine learning?"
contrastive_data = {
    'positive_samples': [...],
    'negative_samples': [...],
    'triplets': [...]
}

# 应用重排序
final_docs = show_retrieved_documents_with_rerank(
    db=vector_db,
    question=question,
    contrastive_data=contrastive_data,
    reranker=simple_reranker
)
```

#### 5.3 批量演示

```python
def demo_reranking_pipeline():
    """完整的重排序演示流程"""
    
    demo_questions = [
        "What is machine learning?",
        "How does AI work?",
        "What are neural networks?"
    ]
    
    for question in demo_questions:
        print(f"\n{'='*60}")
        print(f"演示问题: {question}")
        print('='*60)
        
        # 原始检索
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        
        print("\n原始检索结果:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  [{i}] {doc.page_content[:100]}...")
        
        # 重排序后
        reranked = simple_reranker.rerank_documents(question, docs, top_k=3)
        print("\n重排序后结果:")
        for i, (doc, score) in enumerate(reranked, 1):
            print(f"  [{i}] 分数: {score:.3f} | {doc.page_content[:100]}...")
```

### 6. 性能优化

#### 6.1 计算优化

**缓存机制**
```python
from functools import lru_cache

class CachedReranker(SimpleReranker):
    @lru_cache(maxsize=1000)
    def _extract_keywords(self, text):
        return super()._extract_keywords(text)
```

**批处理优化**
```python
def batch_rerank(self, queries, documents_list):
    """批量重排序"""
    return [self.rerank_documents(q, docs) for q, docs in zip(queries, documents_list)]
```

#### 6.2 算法调优

**动态权重调整**
```python
def adaptive_weights(query_length, doc_count):
    """根据查询和文档数量动态调整权重"""
    if query_length < 5:
        return 0.8, 0.2  # 短查询更重关键词
    else:
        return 0.6, 0.4  # 长查询平衡考虑
```

### 7. 质量分析

#### 7.1 检索质量评估

**质量分数计算**
```python
def analyze_quality(retrieved_docs, contrastive_data):
    positive = count_positive_docs(retrieved_docs, contrastive_data)
    total = len(retrieved_docs)
    
    quality_score = (positive / total) * 100
    
    if quality_score >= 80:
        level = "优秀"
    elif quality_score >= 60:
        level = "良好"
    else:
        level = "需要改进"
    
    return {"score": quality_score, "level": level}
```

#### 7.2 重排序效果分析

**改进度量**
```python
def measure_rerank_improvement(original, reranked, contrastive_data):
    original_quality = analyze_quality(original, contrastive_data)
    reranked_quality = analyze_quality(reranked, contrastive_data)
    
    improvement = reranked_quality['score'] - original_quality['score']
    
    return {
        "original_score": original_quality['score'],
        "reranked_score": reranked_quality['score'],
        "improvement": improvement,
        "significance": "显著提升" if improvement > 10 else "轻微提升"
    }
```

### 8. 扩展功能

#### 8.1 自定义重排序器

**扩展基类**
```python
class CustomReranker(SimpleReranker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_weight = kwargs.get('custom_weight', 0.5)
    
    def _calculate_custom_score(self, query, doc_text):
        """添加自定义评分维度"""
        # 实现自定义评分逻辑
        return 0.0
    
    def rerank_documents(self, query, documents, top_k=5):
        base_results = super().rerank_documents(query, documents, len(documents))
        
        # 添加自定义评分
        enhanced_results = []
        for doc, base_score in base_results:
            custom_score = self._calculate_custom_score(query, doc.page_content)
            final_score = base_score * (1 - self.custom_weight) + custom_score * self.custom_weight
            enhanced_results.append((doc, final_score))
        
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:top_k]
```

#### 8.2 机器学习增强

**集成预训练模型**
```python
class MLLMReranker:
    def __init__(self, model_name="cross-encoder"):
        self.model = load_cross_encoder(model_name)
    
    def rerank_documents(self, query, documents, top_k=5):
        scores = []
        for doc in documents:
            score = self.model.predict([(query, doc.page_content)])[0]
            scores.append((doc, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 9. 监控与调试

#### 9.1 调试工具

**详细日志**
```python
import logging

class DebugReranker(SimpleReranker):
    def rerank_documents(self, query, documents, top_k=5):
        logger.debug(f"Query: {query}")
        logger.debug(f"Documents count: {len(documents)}")
        
        query_keywords = self._extract_keywords(query.lower())
        logger.debug(f"Keywords: {query_keywords}")
        
        results = super().rerank_documents(query, documents, top_k)
        
        for doc, score in results:
            logger.debug(f"Score: {score:.3f}, Doc: {doc.page_content[:100]}...")
        
        return results
```

#### 9.2 性能监控

**指标收集**
```python
class MonitoredReranker:
    def __init__(self, base_reranker):
        self.base_reranker = base_reranker
        self.metrics = {
            'total_queries': 0,
            'total_documents': 0,
            'avg_processing_time': 0.0
        }
    
    def rerank_documents(self, query, documents, top_k=5):
        start_time = time.time()
        
        result = self.base_reranker.rerank_documents(query, documents, top_k)
        
        self.metrics['total_queries'] += 1
        self.metrics['total_documents'] += len(documents)
        processing_time = time.time() - start_time
        
        # 更新平均处理时间
        total_time = self.metrics['avg_processing_time'] * (self.metrics['total_queries'] - 1)
        self.metrics['avg_processing_time'] = (total_time + processing_time) / self.metrics['total_queries']
        
        return result
```

### 10. 部署建议

#### 10.1 生产配置

**推荐参数**
```python
# 生产环境配置
production_reranker = SimpleReranker(
    keyword_weight=0.75,
    length_weight=0.25
)

# 高负载优化
high_performance_reranker = CachedReranker(
    keyword_weight=0.7,
    length_weight=0.3
)
```

#### 10.2 A/B测试框架

```python
def ab_test_rerankers(query, documents, rerankers):
    """A/B测试多个重排序器"""
    results = {}
    for name, reranker in rerankers.items():
        start_time = time.time()
        reranked = reranker.rerank_documents(query, documents, top_k=5)
        processing_time = time.time() - start_time
        
        results[name] = {
            'results': reranked,
            'processing_time': processing_time,
            'top_score': reranked[0][1] if reranked else 0
        }
    
    return results
```

## 总结

本重排序系统提供了完整的RAG检索优化解决方案，通过多层次、多维度的重排序算法显著提升检索质量。系统设计兼顾了性能、准确性和扩展性，支持从简单的启发式方法到复杂的机器学习增强，适用于不同规模和需求的应用场景。通过灵活的配置和丰富的调试工具，开发者可以快速集成并优化其RAG系统的检索效果。