# 检索算法详解与代码实现

## 1. 稠密检索 (Dense Retrieval) - 余弦相似度

### 原理介绍
稠密检索使用神经网络将文档和查询映射到高维向量空间中，通过计算向量间的余弦相似度来衡量相关性。这种方法能够捕获语义相似性，即使文档和查询没有共同词汇也能找到相关内容。

### 核心特点
- **语义理解**：能够理解词汇的语义关系
- **稠密表示**：将文本转换为稠密的实数向量
- **端到端训练**：可以针对特定任务进行优化

### 代码实现

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

class DenseRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        初始化稠密检索器
        Args:
            model_name: 预训练的句子编码器模型名称
        """
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = None
        self.documents = None
    
    def encode_documents(self, documents):
        """
        对文档进行编码
        Args:
            documents: 文档列表
        """
        self.documents = documents
        print(f"正在编码 {len(documents)} 个文档...")
        self.document_embeddings = self.model.encode(
            documents, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
    def search(self, query, top_k=5):
        """
        搜索相关文档
        Args:
            query: 查询字符串
            top_k: 返回最相关的前k个文档
        Returns:
            list: [(文档索引, 相似度分数, 文档内容)]
        """
        if self.document_embeddings is None:
            raise ValueError("请先调用 encode_documents 方法")
        
        # 编码查询
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # 计算余弦相似度
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0), 
            self.document_embeddings, 
            dim=1
        )
        
        # 获取最相关的文档
        top_indices = torch.topk(similarities, k=min(top_k, len(self.documents)))
        
        results = []
        for idx, score in zip(top_indices.indices, top_indices.values):
            results.append((
                idx.item(), 
                score.item(), 
                self.documents[idx.item()]
            ))
        
        return results
    
    def batch_search(self, queries, top_k=5):
        """
        批量搜索
        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数量
        """
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        
        # 计算所有查询与所有文档的相似度
        similarities = torch.cosine_similarity(
            query_embeddings.unsqueeze(1), 
            self.document_embeddings.unsqueeze(0), 
            dim=2
        )
        
        batch_results = []
        for i, query in enumerate(queries):
            top_indices = torch.topk(similarities[i], k=min(top_k, len(self.documents)))
            query_results = []
            for idx, score in zip(top_indices.indices, top_indices.values):
                query_results.append((
                    idx.item(), 
                    score.item(), 
                    self.documents[idx.item()]
                ))
            batch_results.append(query_results)
        
        return batch_results

# 使用示例
if __name__ == "__main__":
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习使用神经网络进行模式识别",
        "自然语言处理帮助计算机理解人类语言",
        "计算机视觉让机器能够理解图像和视频"
    ]
    
    retriever = DenseRetriever()
    retriever.encode_documents(documents)
    
    query = "什么是AI技术"
    results = retriever.search(query, top_k=3)
    
    print(f"查询: {query}")
    for idx, score, doc in results:
        print(f"文档 {idx}: {score:.4f} - {doc}")
```

## 2. 稀疏检索 (Sparse Retrieval) - BM25

### 原理介绍
BM25是基于TF-IDF的改进算法，通过词频(Term Frequency)和逆文档频率(Inverse Document Frequency)来计算文档与查询的相关性。它考虑了词汇的重要性和文档长度的影响。

### 数学公式
```
Score(D,Q) = ∑(i=1 to n) IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))
```

其中：
- f(qi,D)：词qi在文档D中的频率
- |D|：文档D的长度
- avgdl：平均文档长度
- k1, b：调节参数

### 代码实现

```python
import math
import jieba
from collections import defaultdict, Counter
import numpy as np

class BM25Retriever:
    def __init__(self, k1=1.2, b=0.75):
        """
        初始化BM25检索器
        Args:
            k1: 控制词频饱和度的参数
            b: 控制文档长度归一化的参数
        """
        self.k1 = k1
        self.b = b
        self.documents = None
        self.doc_tokens = None
        self.doc_lengths = None
        self.avg_doc_length = 0
        self.idf_values = {}
        self.vocab = set()
        
    def tokenize(self, text):
        """分词函数"""
        return list(jieba.cut(text.lower()))
    
    def build_index(self, documents):
        """
        构建索引
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_tokens = [self.tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # 构建词汇表
        for tokens in self.doc_tokens:
            self.vocab.update(tokens)
        
        # 计算IDF值
        self._calculate_idf()
        
    def _calculate_idf(self):
        """计算每个词的IDF值"""
        n_docs = len(self.documents)
        
        for term in self.vocab:
            # 计算包含该词的文档数量
            doc_freq = sum(1 for tokens in self.doc_tokens if term in tokens)
            # 计算IDF
            self.idf_values[term] = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    
    def _calculate_bm25_score(self, query_tokens, doc_idx):
        """
        计算BM25分数
        Args:
            query_tokens: 查询的分词结果
            doc_idx: 文档索引
        Returns:
            float: BM25分数
        """
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # 计算词频
        term_freq = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term in term_freq:
                tf = term_freq[term]
                idf = self.idf_values.get(term, 0)
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * (numerator / denominator)
                
        return score
    
    def search(self, query, top_k=5):
        """
        搜索相关文档
        Args:
            query: 查询字符串
            top_k: 返回最相关的前k个文档
        Returns:
            list: [(文档索引, BM25分数, 文档内容)]
        """
        if self.documents is None:
            raise ValueError("请先调用 build_index 方法")
        
        query_tokens = self.tokenize(query)
        
        # 计算所有文档的BM25分数
        scores = []
        for doc_idx in range(len(self.documents)):
            score = self._calculate_bm25_score(query_tokens, doc_idx)
            scores.append((doc_idx, score, self.documents[doc_idx]))
        
        # 按分数排序并返回前k个结果
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def explain_score(self, query, doc_idx):
        """
        解释BM25分数的计算过程
        Args:
            query: 查询字符串
            doc_idx: 文档索引
        """
        query_tokens = self.tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        term_freq = Counter(doc_tokens)
        
        print(f"查询: {query}")
        print(f"文档: {self.documents[doc_idx]}")
        print(f"文档长度: {doc_length}, 平均长度: {self.avg_doc_length:.2f}")
        print("-" * 50)
        
        total_score = 0.0
        for term in query_tokens:
            if term in term_freq:
                tf = term_freq[term]
                idf = self.idf_values.get(term, 0)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                term_score = idf * (numerator / denominator)
                total_score += term_score
                
                print(f"词汇: '{term}'")
                print(f"  TF: {tf}, IDF: {idf:.4f}")
                print(f"  分数: {term_score:.4f}")
        
        print(f"总分: {total_score:.4f}")

# 使用示例
if __name__ == "__main__":
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习使用神经网络进行模式识别",
        "自然语言处理帮助计算机理解人类语言",
        "计算机视觉让机器能够理解图像和视频"
    ]
    
    bm25 = BM25Retriever()
    bm25.build_index(documents)
    
    query = "计算机人工智能"
    results = bm25.search(query, top_k=3)
    
    print(f"查询: {query}")
    for idx, score, doc in results:
        print(f"文档 {idx}: {score:.4f} - {doc}")
```

## 3. 混合检索 (Hybrid Retrieval) - 加权融合

### 原理介绍
混合检索结合了稠密检索和稀疏检索的优势，通过加权融合两种方法的结果来提高检索效果。它能够同时利用语义匹配和精确匹配的能力。

### 融合策略
1. **线性加权**：Score = α × Dense_Score + β × Sparse_Score
2. **倒排名融合**：基于排名进行融合
3. **学习融合**：使用机器学习方法学习最优权重

### 代码实现

```python
from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import rankdata

class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, 
                 dense_weight=0.5, sparse_weight=0.5):
        """
        初始化混合检索器
        Args:
            dense_retriever: 稠密检索器
            sparse_retriever: 稀疏检索器
            dense_weight: 稠密检索权重
            sparse_weight: 稀疏检索权重
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
    def normalize_scores(self, scores):
        """归一化分数到[0,1]区间"""
        if len(scores) == 0:
            return scores
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def linear_fusion(self, query, top_k=10):
        """
        线性加权融合
        Args:
            query: 查询字符串
            top_k: 返回结果数量
        Returns:
            list: 融合后的检索结果
        """
        # 获取两种检索方法的结果
        dense_results = self.dense_retriever.search(query, top_k=top_k*2)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k*2)
        
        # 构建分数字典
        dense_scores = {idx: score for idx, score, _ in dense_results}
        sparse_scores = {idx: score for idx, score, _ in sparse_results}
        
        # 归一化分数
        if dense_scores:
            dense_values = self.normalize_scores(list(dense_scores.values()))
            dense_scores = dict(zip(dense_scores.keys(), dense_values))
        
        if sparse_scores:
            sparse_values = self.normalize_scores(list(sparse_scores.values()))
            sparse_scores = dict(zip(sparse_scores.keys(), sparse_values))
        
        # 计算融合分数
        all_doc_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        fused_scores = {}
        
        for doc_idx in all_doc_indices:
            dense_score = dense_scores.get(doc_idx, 0)
            sparse_score = sparse_scores.get(doc_idx, 0)
            fused_score = (self.dense_weight * dense_score + 
                          self.sparse_weight * sparse_score)
            fused_scores[doc_idx] = fused_score
        
        # 排序并返回结果
        sorted_results = sorted(fused_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_idx, score in sorted_results[:top_k]:
            doc_content = self.dense_retriever.documents[doc_idx]
            final_results.append((doc_idx, score, doc_content))
        
        return final_results
    
    def reciprocal_rank_fusion(self, query, top_k=10, k=60):
        """
        倒排名融合 (Reciprocal Rank Fusion)
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            k: RRF参数，用于平滑排名
        Returns:
            list: 融合后的检索结果
        """
        # 获取两种检索方法的结果
        dense_results = self.dense_retriever.search(query, top_k=top_k*2)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k*2)
        
        # 计算RRF分数
        rrf_scores = defaultdict(float)
        
        # 处理稠密检索结果
        for rank, (doc_idx, _, _) in enumerate(dense_results, 1):
            rrf_scores[doc_idx] += self.dense_weight / (k + rank)
        
        # 处理稀疏检索结果
        for rank, (doc_idx, _, _) in enumerate(sparse_results, 1):
            rrf_scores[doc_idx] += self.sparse_weight / (k + rank)
        
        # 排序并返回结果
        sorted_results = sorted(rrf_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_idx, score in sorted_results[:top_k]:
            doc_content = self.dense_retriever.documents[doc_idx]
            final_results.append((doc_idx, score, doc_content))
        
        return final_results
    
    def adaptive_fusion(self, query, top_k=10):
        """
        自适应融合：根据查询特征调整权重
        Args:
            query: 查询字符串
            top_k: 返回结果数量
        """
        # 分析查询特征
        query_tokens = self.sparse_retriever.tokenize(query)
        query_length = len(query_tokens)
        
        # 基于查询长度调整权重
        if query_length <= 2:
            # 短查询更依赖精确匹配
            dense_w, sparse_w = 0.3, 0.7
        elif query_length <= 5:
            # 中等长度查询平衡两种方法
            dense_w, sparse_w = 0.5, 0.5
        else:
            # 长查询更依赖语义匹配
            dense_w, sparse_w = 0.7, 0.3
        
        # 临时调整权重
        original_dense_w = self.dense_weight
        original_sparse_w = self.sparse_weight
        
        self.dense_weight = dense_w
        self.sparse_weight = sparse_w
        
        # 执行融合
        results = self.linear_fusion(query, top_k)
        
        # 恢复原始权重
        self.dense_weight = original_dense_w
        self.sparse_weight = original_sparse_w
        
        return results
    
    def evaluate_fusion_strategies(self, queries, ground_truth=None):
        """
        评估不同融合策略的效果
        Args:
            queries: 查询列表
            ground_truth: 真实相关文档（可选）
        """
        strategies = {
            'linear': self.linear_fusion,
            'rrf': self.reciprocal_rank_fusion,
            'adaptive': self.adaptive_fusion
        }
        
        results = {}
        for strategy_name, strategy_func in strategies.items():
            strategy_results = []
            for query in queries:
                result = strategy_func(query)
                strategy_results.append(result)
            results[strategy_name] = strategy_results
        
        return results

# 使用示例
if __name__ == "__main__":
    # 初始化检索器
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习使用神经网络进行模式识别",
        "自然语言处理帮助计算机理解人类语言",
        "计算机视觉让机器能够理解图像和视频"
    ]
    
    dense_retriever = DenseRetriever()
    dense_retriever.encode_documents(documents)
    
    sparse_retriever = BM25Retriever()
    sparse_retriever.build_index(documents)
    
    # 创建混合检索器
    hybrid = HybridRetriever(dense_retriever, sparse_retriever, 
                            dense_weight=0.6, sparse_weight=0.4)
    
    query = "AI技术应用"
    
    print("=== 线性融合结果 ===")
    linear_results = hybrid.linear_fusion(query, top_k=3)
    for idx, score, doc in linear_results:
        print(f"文档 {idx}: {score:.4f} - {doc}")
    
    print("\n=== RRF融合结果 ===")
    rrf_results = hybrid.reciprocal_rank_fusion(query, top_k=3)
    for idx, score, doc in rrf_results:
        print(f"文档 {idx}: {score:.4f} - {doc}")
```

## 4. 重排序 (Re-ranking) - Cross-encoder

### 原理介绍
重排序是在初步检索结果基础上，使用更精确但计算成本更高的模型对候选文档进行重新排序。Cross-encoder模型同时处理查询和文档，能够捕获更细粒度的相关性信息。

### 架构特点
- **联合编码**：同时编码查询和文档
- **注意力机制**：能够学习查询-文档间的交互
- **高精度**：相比双塔模型有更好的排序效果
- **高延迟**：计算成本较高，适合重排序阶段

### 代码实现

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        初始化Cross-encoder重排序器
        Args:
            model_name: 预训练的cross-encoder模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # 检查是否有GPU可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def score_pairs(self, query, documents, batch_size=16):
        """
        计算查询-文档对的相关性分数
        Args:
            query: 查询字符串
            documents: 文档列表
            batch_size: 批处理大小
        Returns:
            list: 相关性分数列表
        """
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # 构建输入对
                pairs = [[query, doc] for doc in batch_docs]
                
                # 分词和编码
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 获取分数（假设模型输出相关性分数）
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                if len(batch_scores.shape) == 0:
                    batch_scores = [batch_scores.item()]
                else:
                    batch_scores = batch_scores.tolist()
                
                scores.extend(batch_scores)
        
        return scores
    
    def rerank(self, query, candidates, top_k=None):
        """
        重排序候选文档
        Args:
            query: 查询字符串
            candidates: 候选文档列表 [(doc_idx, initial_score, doc_content), ...]
            top_k: 重排序后返回的文档数量
        Returns:
            list: 重排序后的结果 [(doc_idx, rerank_score, doc_content), ...]
        """
        if not candidates:
            return []
        
        # 提取文档内容
        documents = [doc_content for _, _, doc_content in candidates]
        doc_indices = [doc_idx for doc_idx, _, _ in candidates]
        
        # 计算重排序分数
        rerank_scores = self.score_pairs(query, documents)
        
        # 组合结果
        reranked_results = []
        for i, (doc_idx, _, doc_content) in enumerate(candidates):
            reranked_results.append((doc_idx, rerank_scores[i], doc_content))
        
        # 按重排序分数排序
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        return reranked_results
    
    def explain_scores(self, query, documents, show_tokens=False):
        """
        解释重排序分数（简化版本）
        Args:
            query: 查询字符串
            documents: 文档列表
            show_tokens: 是否显示token信息
        """
        scores = self.score_pairs(query, documents)
        
        print(f"查询: {query}")
        print("-" * 50)
        
        for i, (doc, score) in enumerate(zip(documents, scores)):
            print(f"文档 {i}: {score:.4f}")
            print(f"内容: {doc}")
            
            if show_tokens:
                # 显示分词结果
                inputs = self.tokenizer([query, doc], 
                                      return_tensors='pt', 
                                      padding=True, 
                                      truncation=True)
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                print(f"Token: {tokens}")
            
            print("-" * 30)

class HybridRetrieverWithReranking:
    def __init__(self, retriever, reranker, retrieve_k=20, final_k=5):
        """
        带重排序的混合检索系统
        Args:
            retriever: 初步检索器（可以是稠密、稀疏或混合检索器）
            reranker: 重排序器
            retrieve_k: 初步检索返回的候选数量
            final_k: 最终返回的结果数量
        """
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.final_k = final_k
    
    def search(self, query):
        """
        执行两阶段检索：初步检索 + 重排序
        Args:
            query: 查询字符串
        Returns:
            dict: 包含初步检索结果和重排序结果
        """
        # 第一阶段：初步检索
        if hasattr(self.retriever, 'linear_fusion'):
            # 混合检索器
            initial_results = self.retriever.linear_fusion(query, top_k=self.retrieve_k)
        else:
            # 单一检索器
            initial_results = self.retriever.search(query, top_k=self.retrieve_k)
        
        # 第二阶段：重排序
        reranked_results = self.reranker.rerank(
            query, initial_results, top_k=self.final_k
        )
        
        return {
            'initial_results': initial_results,
            'reranked_results': reranked_results,
            'query': query
        }
    
    def evaluate_reranking_effect(self, query):
        """
        评估重排序的效果
        """
        results = self.search(query)
        
        print(f"查询: {query}")
        print("=" * 60)
        
        print("初步检索结果:")
        for i, (idx, score, doc) in enumerate(results['initial_results'][:self.final_k]):
            print(f"{i+1}. [文档{idx}] {score:.4f} - {doc}")
        
        print(f"\n重排序后结果:")
        for i, (idx, score, doc) in enumerate(results['reranked_results']):
            print(f"{i+1}. [文档{idx}] {score:.4f} - {doc}")
        
        # 分析排序变化
        initial_order = [idx for idx, _, _ in results['initial_results'][:self.final_k]]
        reranked_order = [idx for idx, _, _ in results['reranked_results']]
        
        print(f"\n排序变化:")
        print(f"初步检索: {initial_order}")
        print(f"重排序后: {reranked_order}")
        
        # 计算排序变化程度
        position_changes = 0
        for i, doc_idx in enumerate(reranked_order):
            if doc_idx in initial_order:
                old_pos = initial_order.index(doc_idx)
                if old_pos != i:
                    position_changes += 1
        
        print(f"位置变化的文档数量: {position_changes}/{len(reranked_order)}")

# 使用示例
if __name__ == "__main__":
    # 注意：这个示例需要安装transformers库
    # pip install transformers torch
    
    documents = [
        "人工智能是计算机科学的一个分支，专注于创建能够执行通常需要人类智能的任务的系统",
        "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进",
        "深度学习是机器学习的一个分支，使用神经网络进行复杂的模式识别和数据分析",
        "自然语言处理是人工智能的一个领域，专注于使计算机理解、解释和生成人类语言",
        "计算机视觉使机器能够从图像和视频中获取信息，理解视觉世界的内容",
        "强化学习是一种机器学习方法，智能体通过与环境交互来学习最优行为策略",
        "神经网络是一种受生物神经系统启发的计算模型，用于各种机器学习任务"
    ]
    
    # 初始化检索器和重排序器
    dense_retriever = DenseRetriever()
    dense_retriever.encode_documents(documents)
    
    sparse_retriever = BM25Retriever()
    sparse_retriever.build_index(documents)
    
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    
    # 注意：实际使用时需要确保模型可用
    try:
        reranker = CrossEncoderReranker()
        
        # 创建带重排序的系统
        retrieval_system = HybridRetrieverWithReranking(
            hybrid_retriever, reranker, retrieve_k=5, final_k=3
        )
        
        query = "什么是机器学习技术"
        retrieval_system.evaluate_reranking_effect(query)
        
    except Exception as e:
        print(f"重排序器初始化失败: {e}")
        print("请确保已安装transformers库并有网络连接下载模型")
```

## 总结与最佳实践

### 各算法特点对比

| 算法类型 | 优势 | 劣势 | 适用场景 |
|---------|------|------|----------|
| 稠密检索 | 语义理解强，能处理同义词 | 计算成本高，需要训练 | 开放域问答，语义搜索 |
| 稀疏检索 | 速度快，精确匹配好 | 无法理解语义关系 | 关键词搜索，专业术语检索 |
| 混合检索 | 结合两者优势，效果更好 | 复杂度增加 | 通用检索系统 |
| 重排序 | 精度最高 | 延迟高，计算成本大 | 高质量要求的应用 |

### 系统设计建议

1. **分层架构**：使用快速的初步检索缩小候选范围，再用精确的重排序提升质量
2. **参数调优**：根据具体应用场景调整各算法的参数
3. **缓存机制**：对常见查询缓存结果，提升响应速度
4. **评估指标**：使用多种评估指标（如NDCG、MRR、Recall@K）综合评估效果
5. **在线学习**：收集用户反馈，持续优化检索效果

### 实际应用建议

1. **文档预处理**：做好文本清洗、分段、去重等预处理工作
2. **索引优化**：定期更新索引，保持数据新鲜度
3. **监控系统**：监控检索延迟、准确率等关键指标
4. **A/B测试**：通过实验验证算法改进的效果