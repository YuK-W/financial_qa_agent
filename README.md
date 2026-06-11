# 金融长文本智能问答 Agent

金融文档(合同/财报/保险/法规/研报)的智能问答系统，基于 **Qwen3.7-max** API，
支持 A榜(doc_ids) 和 B榜(全领域检索) 双模式。

## 环境要求

- Python 3.10+
- 阿里云百炼 API Key (DashScope)

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key (二选一)
set DASHSCOPE_API_KEY=your_key          # Windows
# 或 创建 .env 文件: DASHSCOPE_API_KEY=your_key

# 3. 验证环境
python test_api.py

# 4. 运行 A榜全量测试
python run_benchmark.py                 # 全量100题
python run_benchmark.py --dry-run       # 快速验证(2题)
```

## 项目结构

```
financial_qa_agent/
├── financial_qa_agent.py   # 核心Agent (A榜/B榜双模式)
├── retrieval_system.py     # BM25+关键词+实体 混合检索
├── domain_prompts.py       # 5领域专用Prompt模板
├── multi_step_reasoning.py # 多轮推理+自检验证
├── numerical_calculator.py # 中文数字解析+数值验算
├── evidence_aggregator.py  # 多文档证据聚合+矛盾检测
├── batch_processor.py      # 批量处理(文档预加载复用)
├── context_optimizer.py    # 自适应上下文窗口
├── cache_manager.py        # LRU缓存管理器
├── pdf_parser.py           # PDF结构化解析
├── chunking.py             # 文档分块器
├── domain_parsers.py       # 领域特定解析(合同/保险/法规)
├── config.py               # 统一配置(路径/Token/检索参数)
├── exceptions.py           # 异常层级定义
├── logger.py               # 集中式日志(loguru)
├── run_benchmark.py        # A榜全量测试+answer.csv生成
├── error_analysis.py       # 错误分析工具
├── test_api.py             # API连通性测试
├── explore_data.py         # 数据探索
├── requirements.txt        # 依赖清单
├── logs/                   # 日志文件(自动生成)
└── data/                   # 比赛数据
    └── public_dataset_upload/
        ├── questions/group_a/  # 5领域100题
        └── raw/                # 原始文档PDF (5领域)
```

## 核心架构

```
题目输入
  │
  ├── A榜: doc_ids → 定位文档
  └── B榜: domain → 全领域检索
  │
  ▼
文档索引(BM25+关键词+实体, 解析阶段不计Token)
  │
  ▼
证据检索 → 证据聚合+跨文档矛盾检测 → 领域专用Prompt
  │
  ▼
Qwen3.7-max 推理 (单轮/多轮自检)
  │
  ▼
答案提取(4层策略) → 数值验算 → 格式校验
  │
  ▼
answer.csv
```

## Token优化策略

| 策略 | 说明 | 效果 |
|------|------|------|
| BM25检索 | 只送top-10相关块, 不送全文 | 省90%+ Token |
| 领域Prompt | 专用分析框架, 减少废话 | 省20%输出Token |
| 文档预索引 | 解析阶段不计Token | 索引完全免费 |
| 批处理复用 | 同文档多题只读一次 | 省I/O+Token |
| LRU缓存 | 高频文档/结果缓存 | 减少重复API调用 |
| 数值验算 | 本地计算, 不消耗API | 省推理Token |

## 提交格式

`answer.csv` 格式:
| qid | answer | prompt_tokens | completion_tokens | total_tokens |
|-----|--------|---------------|-------------------|--------------|
| summary | | 5961 | 5198 | 11159 |
| fc_a_001 | A | 2966 | 1221 | 4187 |
| ... | ... | ... | ... | ... |

## 运行模式

```python
from financial_qa_agent import FinancialQAAgent

# 默认: A榜 + 单轮推理 (省Token)
agent = FinancialQAAgent()

# B榜 + 多轮推理 (更准确)
agent = FinancialQAAgent(use_b_mode=True, use_multi_step=True)
```
