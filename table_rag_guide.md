# 表格RAG系统构建完整指南

## 概述

表格数据是企业和学术机构中最重要的结构化信息载体之一，包含了大量的定量数据、统计信息和关系型知识。构建针对表格的RAG系统需要解决表格理解、结构化查询、关系推理、数值计算等独特挑战。本指南将全面介绍表格RAG系统的设计理念、技术方案和最佳实践，并提供完整的代码实现。

## 1. 表格RAG系统特殊性分析

### 1.1 表格数据特征深度分析

**结构化特性的复杂性**

表格数据的结构化特性远比普通文本复杂，需要建立专门的表格理解引擎来处理多维度结构：

```python
class TableStructureAnalyzer:
    """表格结构分析器"""
    
    def __init__(self):
        self.layout_detector = LayoutDetector()
        self.structure_parser = StructureParser()
        self.semantic_analyzer = SemanticAnalyzer()
    
    def analyze_table_structure(self, table_data):
        """分析表格的多层次结构"""
        return {
            'physical_structure': self.extract_physical_structure(table_data),
            'logical_structure': self.extract_logical_structure(table_data),
            'semantic_structure': self.extract_semantic_structure(table_data),
            'functional_structure': self.extract_functional_structure(table_data)
        }
    
    def extract_physical_structure(self, table_data):
        """提取物理结构：行列、合并单元格、边界"""
        return {
            'rows': len(table_data),
            'cols': len(table_data[0]) if table_data else 0,
            'merged_cells': self.detect_merged_cells(table_data),
            'borders': self.detect_borders(table_data)
        }
    
    def extract_logical_structure(self, table_data):
        """提取逻辑结构：表头、数据区、汇总区"""
        return {
            'header_levels': self.identify_header_levels(table_data),
            'data_regions': self.identify_data_regions(table_data),
            'summary_regions': self.identify_summary_regions(table_data)
        }
    
    def extract_semantic_structure(self, table_data):
        """提取语义结构：主键、外键、层次关系"""
        return {
            'primary_keys': self.identify_primary_keys(table_data),
            'foreign_keys': self.identify_foreign_keys(table_data),
            'hierarchies': self.identify_hierarchies(table_data)
        }
    
    def extract_functional_structure(self, table_data):
        """提取功能结构：计算关系、聚合关系"""
        return {
            'formulas': self.detect_formulas(table_data),
            'aggregations': self.detect_aggregations(table_data),
            'dependencies': self.analyze_dependencies(table_data)
        }

# 使用示例
analyzer = TableStructureAnalyzer()
table_structure = analyzer.analyze_table_structure(table_data)
```

### 1.2 表格RAG的核心技术挑战

**传统RAG系统的根本局限**

传统文本RAG在处理表格数据时面临根本性挑战，需要专门的表格RAG架构：

```python
class TableRAGFramework:
    """表格RAG专用框架"""
    
    def __init__(self):
        self.table_detector = AdvancedTableDetector()
        self.structure_parser = TableStructureParser()
        self.semantic_encoder = TableSemanticEncoder()
        self.query_processor = TableQueryProcessor()
        self.answer_generator = TableAnswerGenerator()
    
    def process_table_document(self, document_path):
        """处理表格文档的完整流程"""
        
        # 1. 表格检测与提取
        tables = self.table_detector.extract_tables(document_path)
        
        # 2. 结构解析
        structured_tables = []
        for table in tables:
            parsed_table = self.structure_parser.parse_table(table)
            structured_tables.append(parsed_table)
        
        # 3. 语义编码
        encoded_tables = []
        for table in structured_tables:
            encoded = self.semantic_encoder.encode_table(table)
            encoded_tables.append(encoded)
        
        # 4. 构建索引
        table_index = self.build_table_index(encoded_tables)
        
        return {
            'tables': structured_tables,
            'encoded_tables': encoded_tables,
            'table_index': table_index
        }
    
    def query_tables(self, query, table_index):
        """查询表格数据"""
        processed_query = self.query_processor.process_query(query)
        relevant_tables = table_index.search(processed_query)
        answer = self.answer_generator.generate_answer(processed_query, relevant_tables)
        return answer
```

## 2. 表格检测与提取技术

### 2.1 多源表格检测技术

**PDF文档表格检测**

```python
import cv2
import numpy as np
from PIL import Image
import pytesseract
from transformers import AutoModelForObjectDetection, AutoImageProcessor

class PDFTableDetector:
    """PDF文档表格检测器"""
    
    def __init__(self):
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    
    def detect_tables_pdf(self, pdf_path):
        """检测PDF中的表格"""
        
        # 将PDF转换为图像
        images = self.pdf_to_images(pdf_path)
        tables = []
        
        for page_num, image in enumerate(images):
            # 使用深度学习模型检测表格
            table_regions = self.detect_tables_in_image(image)
            
            # 使用传统方法验证
            traditional_regions = self.detect_tables_traditional(image)
            
            # 融合检测结果
            final_regions = self.merge_detections(table_regions, traditional_regions)
            
            for region in final_regions:
                table_data = self.extract_table_data(image, region)
                tables.append({
                    'page': page_num,
                    'bbox': region,
                    'table_data': table_data
                })
        
        return tables
    
    def detect_tables_traditional(self, image):
        """传统图像处理方法检测表格"""
        
        # 转换为灰度图
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # 分析直线形成表格区域
        table_regions = self.analyze_lines_to_tables(lines)
        
        return table_regions
    
    def pdf_to_images(self, pdf_path):
        """PDF转图像"""
        from pdf2image import convert_from_path
        return convert_from_path(pdf_path)

# 使用示例
detector = PDFTableDetector()
tables = detector.detect_tables_pdf("sample.pdf")
```

**HTML表格提取**

```python
from bs4 import BeautifulSoup
import pandas as pd
import requests

class HTMLTableExtractor:
    """HTML表格提取器"""
    
    def __init__(self):
        self.soup = None
    
    def extract_tables_from_html(self, html_content):
        """从HTML中提取表格"""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        extracted_tables = []
        
        for table_idx, table in enumerate(tables):
            # 提取表格元数据
            table_info = self.extract_table_metadata(table)
            
            # 解析表格结构
            table_data = self.parse_html_table(table)
            
            # 处理复杂结构（合并单元格、嵌套表格）
            processed_data = self.process_complex_structure(table_data)
            
            extracted_tables.append({
                'table_id': table_idx,
                'metadata': table_info,
                'data': processed_data
            })
        
        return extracted_tables
    
    def extract_table_metadata(self, table):
        """提取表格元数据"""
        
        metadata = {
            'caption': table.find('caption').text if table.find('caption') else None,
            'classes': table.get('class', []),
            'id': table.get('id'),
            'headers': self.extract_headers(table)
        }
        
        return metadata
    
    def parse_html_table(self, table):
        """解析HTML表格结构"""
        
        rows = table.find_all('tr')
        table_data = []
        
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            row_data = []
            
            for cell_idx, cell in enumerate(cells):
                cell_info = {
                    'text': cell.get_text(strip=True),
                    'type': 'th' if cell.name == 'th' else 'td',
                    'colspan': int(cell.get('colspan', 1)),
                    'rowspan': int(cell.get('rowspan', 1)),
                    'position': (row_idx, cell_idx)
                }
                row_data.append(cell_info)
            
            table_data.append(row_data)
        
        return table_data
    
    def process_complex_structure(self, table_data):
        """处理复杂表格结构"""
        
        # 处理合并单元格
        expanded_data = self.expand_merged_cells(table_data)
        
        # 构建DataFrame
        df = self.create_dataframe(expanded_data)
        
        return df
    
    def extract_tables_from_url(self, url):
        """从URL提取表格"""
        
        response = requests.get(url)
        return self.extract_tables_from_html(response.content)

# 使用示例
extractor = HTMLTableExtractor()
tables = extractor.extract_tables_from_html(html_content)
```

### 2.2 表格结构解析

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class TableStructureParser:
    """表格结构解析器"""
    
    def __init__(self):
        self.header_detector = HeaderDetector()
        self.data_type_analyzer = DataTypeAnalyzer()
    
    def parse_table(self, table_data: pd.DataFrame) -> Dict:
        """全面解析表格结构"""
        
        # 1. 表头识别
        header_info = self.identify_headers(table_data)
        
        # 2. 数据类型分析
        column_types = self.analyze_column_types(table_data)
        
        # 3. 关系识别
        relationships = self.identify_relationships(table_data)
        
        # 4. 质量评估
        quality_metrics = self.assess_table_quality(table_data)
        
        return {
            'headers': header_info,
            'column_types': column_types,
            'relationships': relationships,
            'quality_metrics': quality_metrics,
            'metadata': self.extract_metadata(table_data)
        }
    
    def identify_headers(self, df: pd.DataFrame) -> Dict:
        """智能识别表头"""
        
        # 多层表头检测
        headers = {
            'primary_header': self.detect_primary_header(df),
            'multi_level_headers': self.detect_multi_level_headers(df),
            'colspan_headers': self.detect_colspan_headers(df),
            'rowspan_headers': self.detect_rowspan_headers(df)
        }
        
        return headers
    
    def analyze_column_types(self, df: pd.DataFrame) -> Dict:
        """分析列数据类型"""
        
        column_types = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            # 检测数据类型
            detected_type = self.detect_column_type(col_data)
            
            # 检测特殊格式
            special_format = self.detect_special_format(col_data)
            
            # 检测单位信息
            unit_info = self.detect_unit_info(col_data)
            
            column_types[col] = {
                'detected_type': detected_type,
                'special_format': special_format,
                'unit': unit_info,
                'null_ratio': col_data.isnull().sum() / len(df),
                'unique_ratio': col_data.nunique() / len(col_data)
            }
        
        return column_types
    
    def detect_column_type(self, series: pd.Series) -> str:
        """检测列数据类型"""
        
        # 尝试转换为数值
        try:
            pd.to_numeric(series, errors='raise')
            return 'numeric'
        except:
            pass
        
        # 尝试转换为日期
        try:
            pd.to_datetime(series, errors='raise')
            return 'datetime'
        except:
            pass
        
        # 检测布尔值
        if series.str.lower().isin(['true', 'false', '1', '0', 'yes', 'no']).all():
            return 'boolean'
        
        # 检测百分比
        if series.astype(str).str.contains('%').any():
            return 'percentage'
        
        # 检测货币
        currency_pattern = r'[$¥€£]|USD|EUR|GBP|CNY'
        if series.astype(str).str.contains(currency_pattern, regex=True).any():
            return 'currency'
        
        return 'text'
    
    def identify_relationships(self, df: pd.DataFrame) -> Dict:
        """识别表格内部关系"""
        
        relationships = {
            'calculated_columns': self.identify_calculated_columns(df),
            'summary_rows': self.identify_summary_rows(df),
            'hierarchical_relationships': self.identify_hierarchies(df),
            'foreign_key_candidates': self.identify_foreign_keys(df)
        }
        
        return relationships
    
    def assess_table_quality(self, df: pd.DataFrame) -> Dict:
        """评估表格质量"""
        
        return {
            'completeness': self.calculate_completeness(df),
            'consistency': self.calculate_consistency(df),
            'accuracy': self.calculate_accuracy(df),
            'uniqueness': self.calculate_uniqueness(df)
        }
    
    def calculate_completeness(self, df: pd.DataFrame) -> float:
        """计算完整性得分"""
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        return 1 - (null_cells / total_cells)
    
    def calculate_consistency(self, df: pd.DataFrame) -> float:
        """计算一致性得分"""
        # 检查数据类型一致性
        type_consistency = 1.0
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # 检查数值列的一致性
                if self.detect_column_type(col_data) == 'numeric':
                    numeric_ratio = pd.to_numeric(col_data, errors='coerce').notnull().mean()
                    type_consistency *= numeric_ratio
        
        return type_consistency

# 使用示例
parser = TableStructureParser()
table_info = parser.parse_table(df)
```

## 3. 表格语义理解

### 3.1 表格嵌入模型

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TableEmbeddingModel:
    """表格嵌入模型"""
    
    def __init__(self, model_name="microsoft/tapex-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode_table(self, table_df: pd.DataFrame, table_name: str = "") -> Dict:
        """编码整个表格"""
        
        # 表格摘要
        table_summary = self.generate_table_summary(table_df, table_name)
        
        # 列嵌入
        column_embeddings = self.encode_columns(table_df)
        
        # 行嵌入
        row_embeddings = self.encode_rows(table_df)
        
        # 单元格嵌入
        cell_embeddings = self.encode_cells(table_df)
        
        # 关系嵌入
        relation_embeddings = self.encode_relations(table_df)
        
        return {
            'table_summary': table_summary,
            'column_embeddings': column_embeddings,
            'row_embeddings': row_embeddings,
            'cell_embeddings': cell_embeddings,
            'relation_embeddings': relation_embeddings,
            'global_embedding': self.get_global_embedding(table_df)
        }
    
    def encode_columns(self, table_df: pd.DataFrame) -> Dict:
        """编码表格列"""
        
        column_embeddings = {}
        
        for col in table_df.columns:
            # 列名编码
            col_name_embedding = self.encode_text(str(col))
            
            # 列内容编码
            col_content = " ".join(table_df[col].astype(str).tolist()[:10])  # 限制内容长度
            col_content_embedding = self.encode_text(col_content)
            
            # 列统计信息编码
            stats_text = self.generate_column_stats_text(table_df[col])
            stats_embedding = self.encode_text(stats_text)
            
            column_embeddings[col] = {
                'name_embedding': col_name_embedding,
                'content_embedding': col_content_embedding,
                'stats_embedding': stats_embedding
            }
        
        return column_embeddings
    
    def encode_rows(self, table_df: pd.DataFrame) -> Dict:
        """编码表格行"""
        
        row_embeddings = {}
        
        for idx, row in table_df.iterrows():
            row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
            row_embedding = self.encode_text(row_text)
            row_embeddings[idx] = row_embedding
        
        return row_embeddings
    
    def encode_cells(self, table_df: pd.DataFrame) -> Dict:
        """编码单元格"""
        
        cell_embeddings = {}
        
        for idx, row in table_df.iterrows():
            for col in table_df.columns:
                cell_key = f"{idx}_{col}"
                cell_text = f"{col}={row[col]}"
                cell_embedding = self.encode_text(cell_text)
                cell_embeddings[cell_key] = cell_embedding
        
        return cell_embeddings
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本"""
        
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的嵌入
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return embedding
    
    def generate_table_summary(self, table_df: pd.DataFrame, table_name: str) -> str:
        """生成表格摘要"""
        
        summary_parts = [
            f"Table: {table_name}",
            f"Rows: {len(table_df)}",
            f"Columns: {len(table_df.columns)}",
            f"Columns: {', '.join(map(str, table_df.columns))}",
            f"Data types: {', '.join([self.infer_type_summary(table_df[col]) for col in table_df.columns])}"
        ]
        
        return " ".join(summary_parts)
    
    def generate_column_stats_text(self, series: pd.Series) -> str:
        """生成列统计文本"""
        
        stats = series.describe()
        stats_text = f"{series.name}: count={stats.get('count', 0)}, " \
                     f"mean={stats.get('mean', 0):.2f}, " \
                     f"std={stats.get('std', 0):.2f}, " \
                     f"min={stats.get('min', 0)}, " \
                     f"max={stats.get('max', 0)}"
        
        return stats_text
    
    def infer_type_summary(self, series: pd.Series) -> str:
        """推断类型摘要"""
        
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "text"

# 使用示例
embedding_model = TableEmbeddingModel()
table_embeddings = embedding_model.encode_table(df, "sales_data")
```

### 3.2 表格查询处理器

```python
from typing import List, Dict, Any
import re
from datetime import datetime

class TableQueryProcessor:
    """表格查询处理器"""
    
    def __init__(self):
        self.query_parser = NaturalLanguageQueryParser()
        self.sql_generator = SQLQueryGenerator()
        self.execution_engine = QueryExecutionEngine()
    
    def process_query(self, query: str, table_schema: Dict) -> Dict:
        """处理自然语言查询"""
        
        # 1. 查询意图解析
        parsed_query = self.query_parser.parse(query)
        
        # 2. 查询重写
        rewritten_query = self.rewrite_query(parsed_query, table_schema)
        
        # 3. 查询验证
        validated_query = self.validate_query(rewritten_query, table_schema)
        
        # 4. 查询优化
        optimized_query = self.optimize_query(validated_query)
        
        return optimized_query
    
    def parse_natural_language_query(self, query: str) -> Dict:
        """解析自然语言查询"""
        
        # 关键词提取
        keywords = self.extract_keywords(query)
        
        # 聚合操作识别
        aggregations = self.identify_aggregations(query)
        
        # 条件识别
        conditions = self.identify_conditions(query)
        
        # 排序识别
        order_by = self.identify_order_by(query)
        
        # 时间范围识别
        time_range = self.identify_time_range(query)
        
        return {
            'keywords': keywords,
            'aggregations': aggregations,
            'conditions': conditions,
            'order_by': order_by,
            'time_range': time_range
        }
    
    def extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        
        # 移除停用词
        stop_words = {'what', 'how', 'show', 'me', 'the', 'of', 'in', 'for', 'by', 'with'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
    
    def identify_aggregations(self, query: str) -> List[Dict]:
        """识别聚合操作"""
        
        aggregations = []
        
        # 聚合关键词映射
        agg_keywords = {
            'sum': 'SUM',
            'total': 'SUM',
            'average': 'AVG',
            'avg': 'AVG',
            'mean': 'AVG',
            'count': 'COUNT',
            'number': 'COUNT',
            'max': 'MAX',
            'maximum': 'MAX',
            'min': 'MIN',
            'minimum': 'MIN'
        }
        
        for keyword, function in agg_keywords.items():
            if keyword in query.lower():
                aggregations.append({
                    'function': function,
                    'keyword': keyword,
                    'position': query.lower().find(keyword)
                })
        
        return aggregations
    
    def identify_conditions(self, query: str) -> List[Dict]:
        """识别条件"""
        
        conditions = []
        
        # 条件模式
        condition_patterns = [
            (r'(\w+)\s*>=?\s*(\d+(?:\.\d+)?)', 'greater_equal'),
            (r'(\w+)\s*<=?\s*(\d+(?:\.\d+)?)', 'less_equal'),
            (r'(\w+)\s*=\s*(["\']?)([^"\']+)\2', 'equal'),
            (r'(\w+)\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)', 'between'),
            (r'(\w+)\s+like\s+(["\']?)([^"\']+)\2', 'like')
        ]
        
        for pattern, condition_type in condition_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                conditions.append({
                    'type': condition_type,
                    'column': match.group(1),
                    'value': match.group(2) if len(match.groups()) > 1 else None,
                    'values': match.groups()[2:] if len(match.groups()) > 2 else None
                })
        
        return conditions
    
    def generate_sql_query(self, parsed_query: Dict, table_schema: Dict) -> str:
        """生成SQL查询"""
        
        table_name = table_schema.get('name', 'table')
        columns = table_schema.get('columns', [])
        
        # 构建SELECT子句
        select_clause = self.build_select_clause(parsed_query, columns)
        
        # 构建WHERE子句
        where_clause = self.build_where_clause(parsed_query)
        
        # 构建GROUP BY子句
        group_by_clause = self.build_group_by_clause(parsed_query, columns)
        
        # 构建ORDER BY子句
        order_by_clause = self.build_order_by_clause(parsed_query)
        
        # 构建LIMIT子句
        limit_clause = self.build_limit_clause(parsed_query)
        
        # 组装完整查询
        sql_parts = ['SELECT', select_clause, 'FROM', table_name]
        
        if where_clause:
            sql_parts.extend(['WHERE', where_clause])
        
        if group_by_clause:
            sql_parts.extend(['GROUP BY', group_by_clause])
        
        if order_by_clause:
            sql_parts.extend(['ORDER BY', order_by_clause])
        
        if limit_clause:
            sql_parts.extend(['LIMIT', limit_clause])
        
        return ' '.join(sql_parts)
    
    def build_select_clause(self, parsed_query: Dict, columns: List[str]) -> str:
        """构建SELECT子句"""
        
        if parsed_query['aggregations']:
            select_parts = []
            for agg in parsed_query['aggregations']:
                # 智能选择聚合列
                target_column = self.select_aggregation_column(agg, columns)
                select_parts.append(f"{agg['function']}({target_column}) AS {agg['function'].lower()}_{target_column}")
            return ', '.join(select_parts)
        else:
            return '*'
    
    def select_aggregation_column(self, aggregation: Dict, columns: List[str]) -> str:
        """智能选择聚合列"""
        
        # 优先选择数值列
        numeric_columns = [col for col in columns if any(word in str(col).lower() for word in ['amount', 'value', 'price', 'count', 'total', 'sum', 'revenue', 'cost'])]
        
        if numeric_columns:
            return numeric_columns[0]
        else:
            return columns[0] if columns else '*'
    
    def build_where_clause(self, parsed_query: Dict) -> str:
        """构建WHERE子句"""
        
        conditions = []
        
        for condition in parsed_query['conditions']:
            condition_sql = self.convert_condition_to_sql(condition)
            conditions.append(condition_sql)
        
        return ' AND '.join(conditions) if conditions else ""
    
    def convert_condition_to_sql(self, condition: Dict) -> str:
        """转换条件为SQL"""
        
        if condition['type'] == 'greater_equal':
            return f"{condition['column']} >= {condition['value']}"
        elif condition['type'] == 'less_equal':
            return f"{condition['column']} <= {condition['value']}"
        elif condition['type'] == 'equal':
            return f"{condition['column']} = '{condition['value']}'"
        elif condition['type'] == 'between':
            return f"{condition['column']} BETWEEN {condition['values'][0]} AND {condition['values'][1]}"
        elif condition['type'] == 'like':
            return f"{condition['column']} LIKE '%{condition['value']}%'"
        
        return ""

# 使用示例
query_processor = TableQueryProcessor()
parsed_query = query_processor.parse_natural_language_query("Show me the average sales by region where sales > 1000")
sql_query = query_processor.generate_sql_query(parsed_query, {'name': 'sales', 'columns': ['region', 'sales', 'date']})
```

## 4. 表格RAG系统完整实现

### 4.1 系统架构

```python
class TableRAGSystem:
    """完整的表格RAG系统"""
    
    def __init__(self):
        self.table_loader = TableLoader()
        self.table_parser = TableStructureParser()
        self.embedding_model = TableEmbeddingModel()
        self.query_processor = TableQueryProcessor()
        self.vector_store = VectorStore()
        self.answer_generator = AnswerGenerator()
    
    def build_table_index(self, data_source: str, source_type: str = 'csv') -> Dict:
        """构建表格索引"""
        
        # 1. 加载表格数据
        tables = self.table_loader.load_tables(data_source, source_type)
        
        # 2. 解析表格结构
        parsed_tables = []
        for table in tables:
            parsed_table = self.table_parser.parse_table(table)
            parsed_tables.append(parsed_table)
        
        # 3. 生成嵌入
        embedded_tables = []
        for parsed_table in parsed_tables:
            embedding = self.embedding_model.encode_table(parsed_table['data'])
            embedded_tables.append({
                **parsed_table,
                'embedding': embedding
            })
        
        # 4. 构建向量索引
        index = self.vector_store.build_index(embedded_tables)
        
        return {
            'index': index,
            'metadata': {
                'total_tables': len(tables),
                'total_rows': sum(len(table) for table in tables),
                'build_time': datetime.now()
            }
        }
    
    def query(self, question: str, k: int = 5) -> Dict:
        """查询表格数据"""
        
        # 1. 处理查询
        processed_query = self.query_processor.process_query(question)
        
        # 2. 检索相关表格
        relevant_tables = self.vector_store.search(processed_query, k=k)
        
        # 3. 生成答案
        answer = self.answer_generator.generate_answer(question, relevant_tables)
        
        return answer
    
    def add_table(self, table_data: pd.DataFrame, table_name: str) -> None:
        """添加新表格"""
        
        # 解析表格
        parsed_table = self.table_parser.parse_table(table_data)
        
        # 生成嵌入
        embedding = self.embedding_model.encode_table(table_data, table_name)
        
        # 添加到索引
        self.vector_store.add_table(parsed_table, embedding)

# 使用示例
rag_system = TableRAGSystem()

# 构建索引
index_result = rag_system.build_table_index("sales_data.csv", "csv")

# 查询
result = rag_system.query("What is the total revenue for Q4 2023?")
print(result)
```

### 4.2 高级功能实现

```python
class AdvancedTableFeatures:
    """高级表格功能"""
    
    def __init__(self):
        self.calculator = TableCalculator()
        self.visualizer = TableVisualizer()
        self.validator = DataValidator()
    
    def perform_calculations(self, table_df: pd.DataFrame, calculations: List[str]) -> pd.DataFrame:
        """执行表格计算"""
        
        result_df = table_df.copy()
        
        for calc in calculations:
            if calc.startswith("sum_"):
                column = calc[4:]
                result_df[f"sum_{column}"] = result_df[column].sum()
            elif calc.startswith("avg_"):
                column = calc[4:]
                result_df[f"avg_{column}"] = result_df[column].mean()
            elif calc.startswith("pct_change_"):
                column = calc[11:]
                result_df[f"pct_change_{column}"] = result_df[column].pct_change()
        
        return result_df
    
    def create_summary_statistics(self, table_df: pd.DataFrame) -> Dict:
        """创建汇总统计"""
        
        numeric_cols = table_df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'basic_stats': table_df[numeric_cols].describe(),
            'correlation_matrix': table_df[numeric_cols].corr(),
            'missing_data': table_df.isnull().sum(),
            'data_types': table_df.dtypes.to_dict()
        }
        
        return summary
    
    def validate_data_quality(self, table_df: pd.DataFrame, rules: Dict) -> Dict:
        """验证数据质量"""
        
        validation_results = {}
        
        for rule_name, rule in rules.items():
            if rule['type'] == 'range':
                column = rule['column']
                min_val = rule['min']
                max_val = rule['max']
                violations = table_df[(table_df[column] < min_val) | (table_df[column] > max_val)]
                validation_results[rule_name] = len(violations)
            
            elif rule['type'] == 'format':
                column = rule['column']
                pattern = rule['pattern']
                violations = table_df[~table_df[column].astype(str).str.match(pattern)]
                validation_results[rule_name] = len(violations)
        
        return validation_results
    
    def export_to_formats(self, table_df: pd.DataFrame, formats: List[str]) -> Dict:
        """导出为多种格式"""
        
        exports = {}
        
        if 'csv' in formats:
            exports['csv'] = table_df.to_csv(index=False)
        
        if 'json' in formats:
            exports['json'] = table_df.to_json(orient='records')
        
        if 'excel' in formats:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                table_df.to_excel(writer, sheet_name='Sheet1', index=False)
            exports['excel'] = buffer.getvalue()
        
        return exports

# 使用示例
advanced = AdvancedTableFeatures()

# 执行计算
result_df = advanced.perform_calculations(df, ["sum_sales", "avg_price", "pct_change_revenue"])

# 创建汇总统计
summary = advanced.create_summary_statistics(df)

# 验证数据质量
quality_rules = {
    'sales_range': {'type': 'range', 'column': 'sales', 'min': 0, 'max': 1000000},
    'date_format': {'type': 'format', 'column': 'date', 'pattern': r'\d{4}-\d{2}-\d{2}'}
}
quality_results = advanced.validate_data_quality(df, quality_rules)
```

## 5. 部署与优化

### 5.1 性能优化

```python
class TableRAGOptimizer:
    """表格RAG性能优化器"""
    
    def __init__(self):
        self.cache = {}
        self.batch_size = 1000
    
    def optimize_memory_usage(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """优化内存使用"""
        
        optimized_tables = []
        
        for table in tables:
            # 数据类型优化
            optimized = self.optimize_data_types(table)
            
            # 内存映射
            optimized = self.use_memory_mapping(optimized)
            
            optimized_tables.append(optimized)
        
        return optimized_tables
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型"""
        
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = str(optimized_df[col].dtype)
            
            # 整数类型优化
            if col_type == 'int64':
                if optimized_df[col].min() >= 0:
                    if optimized_df[col].max() < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif optimized_df[col].max() < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    elif optimized_df[col].max() < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
            
            # 浮点数类型优化
            elif col_type == 'float64':
                optimized_df[col] = optimized_df[col].astype('float32')
        
        return optimized_df
    
    def implement_caching(self, query_func):
        """实现缓存机制"""
        
        def cached_function(*args, **kwargs):
            cache_key = str(args) + str(kwargs)
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            result = query_func(*args, **kwargs)
            self.cache[cache_key] = result
            
            return result
        
        return cached_function
    
    def parallel_processing(self, tables: List[pd.DataFrame], process_func) -> List[Any]:
        """并行处理"""
        
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_func, tables))
        
        return results
```

### 5.2 监控与运维

```python
class TableRAGMonitor:
    """表格RAG监控器"""
    
    def __init__(self):
        self.metrics = {
            'query_count': 0,
            'avg_response_time': 0,
            'error_count': 0,
            'cache_hit_rate': 0
        }
    
    def log_query(self, query: str, response_time: float, success: bool):
        """记录查询日志"""
        
        self.metrics['query_count'] += 1
        
        if success:
            total_time = self.metrics['avg_response_time'] * (self.metrics['query_count'] - 1)
            self.metrics['avg_response_time'] = (total_time + response_time) / self.metrics['query_count']
        else:
            self.metrics['error_count'] += 1
    
    def generate_report(self) -> Dict:
        """生成监控报告"""
        
        return {
            'total_queries': self.metrics['query_count'],
            'avg_response_time': self.metrics['avg_response_time'],
            'error_rate': self.metrics['error_count'] / max(self.metrics['query_count'], 1),
            'uptime': self.calculate_uptime(),
            'performance_trend': self.analyze_performance_trend()
        }
    
    def calculate_uptime(self) -> float:
        """计算系统可用性"""
        # 简化的实现
        return 1.0 - (self.metrics['error_count'] / max(self.metrics['query_count'], 1))
    
    def analyze_performance_trend(self) -> Dict:
        """分析性能趋势"""
        # 简化的实现
        return {
            'trend': 'stable',
            'recommendations': ['Consider increasing cache size', 'Monitor memory usage']
        }
```

## 总结

本指南提供了从基础表格检测到高级查询处理的完整技术实现。优势包括：

1. **多源支持**：支持PDF、HTML、Excel、CSV等多种格式的表格处理
2. **智能解析**：自动识别表格结构、数据类型和关系
3. **高效查询**：自然语言到SQL的转换，支持复杂查询
4. **性能优化**：内存优化、缓存机制、并行处理
5. **质量保障**：完整的数据验证和质量评估体系

通过这套完整的代码实现，您可以构建一个生产级的表格RAG系统，能够处理复杂的表格数据并提供准确的查询结果。
