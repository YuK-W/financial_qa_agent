# config.py
"""
项目统一配置文件

所有可调参数、路径、API 设置集中管理于此。
修改配置只需改这一个文件，无需在多个模块间跳转。

使用方式：
    from config import config
    api_key = config.api_key
"""

import os
import sys


class Config:
    """金融问答 Agent 全局配置"""

    # ================================================================
    # API 配置
    # ================================================================
    @property
    def api_key(self) -> str:
        """
        从环境变量安全读取 API Key，绝不硬编码。
        优先级：环境变量 > .env 文件
        """
        # 先尝试手动加载 .env（无需 python-dotenv 依赖）
        self._load_dotenv()

        key = os.getenv("DASHSCOPE_API_KEY", "")
        if not key:
            print("[ERROR] DASHSCOPE_API_KEY not found")
            print("   Set env: set DASHSCOPE_API_KEY=your_key")
            print("   Or create .env file: DASHSCOPE_API_KEY=your_key")
            sys.exit(1)
        return key

    @property
    def model_name(self) -> str:
        """Qwen 模型名称"""
        return "qwen-plus"  # qwen3.6-plus

    # ================================================================
    # 路径配置 —— 基于项目根目录，消除硬编码
    # ================================================================
    @property
    def project_root(self) -> str:
        """项目根目录（config.py 所在目录）"""
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, "data", "public_dataset_upload")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def questions_dir(self) -> str:
        return os.path.join(self.data_dir, "questions")

    # ================================================================
    # 5 个领域 —— 与数据目录实际结构对应
    # ================================================================
    DOMAINS = [
        "financial_contracts",   # 金融合同
        "financial_reports",     # 财务报告
        "insurance",             # 保险条款
        "regulatory",            # 监管法规
        "research",              # 研究报告
    ]

    # 领域中文名映射
    DOMAIN_NAMES = {
        "financial_contracts": "金融合同",
        "financial_reports": "财务报告",
        "insurance": "保险条款",
        "regulatory": "监管法规",
        "research": "研究报告",
    }

    # ================================================================
    # Token / 上下文限制
    # ================================================================
    MAX_CONTEXT_CHARS = 5000          # 上下文最大字符数
    MAX_CONTEXT_TOKENS = 3000         # 上下文最大 token 数（估算）
    MAX_OUTPUT_TOKENS = 800           # 模型输出最大 token 数（默认值）
    MAX_PDF_PAGES = 0                 # 单文档最多读取页数（0=不限，解析阶段不计Token）
    TOKEN_ESTIMATE_RATIO = 1.8        # 中文混合文本：约 1.8 字符/token

    # ================================================================
    # P3-3: 自适应 Token 预算分配
    # ================================================================
    # 基础预算（按题型）
    TOKEN_BUDGET_BASE = {
        'tf': 2000,      # 判断题: 二选一，最简单
        'mcq': 3500,     # 单选题: 四选一
        'multi': 5000,   # 多选题: 需逐一判断
    }

    # 复杂度调整因子
    TOKEN_BUDGET_QUESTION_LENGTH_FACTOR = 0.5   # 问题每100字符加50 token
    TOKEN_BUDGET_DOC_COUNT_FACTOR = 300         # 每多一个文档加300 token
    TOKEN_BUDGET_HAS_CALCULATION = 800          # 涉及计算额外加800 token
    TOKEN_BUDGET_MIN = 1500                     # 最低预算
    TOKEN_BUDGET_MAX = 6000                     # 最高预算上限

    @classmethod
    def adaptive_tokens(cls, answer_format: str, question: str = '',
                         doc_count: int = 1) -> int:
        """
        P3-3: 根据题目难度自适应计算 Token 预算。

        公式:
          base(answer_format)
          + len(question)/100 * QUESTION_LENGTH_FACTOR
          + doc_count * DOC_COUNT_FACTOR
          + HAS_CALCULATION (if 计算/比例/金额/增长率 in question)

        Args:
            answer_format: 'mcq' / 'tf' / 'multi'
            question: 问题文本（用于长度和关键词检测）
            doc_count: 涉及的文档数量

        Returns:
            自适应输出 token 预算（限制在 MIN~MAX 之间）
        """
        base = cls.TOKEN_BUDGET_BASE.get(answer_format, 3500)

        # 问题长度加成
        q_len = len(question) if question else 0
        length_bonus = int(q_len / 100 * cls.TOKEN_BUDGET_QUESTION_LENGTH_FACTOR)

        # 文档数量加成
        doc_bonus = doc_count * cls.TOKEN_BUDGET_DOC_COUNT_FACTOR

        # 计算类题目加成
        calc_keywords = ['计算', '比例', '金额', '亿元', '万元', '%', '增长', '下降',
                         '总额', '利率', '多少', '等于', '合计', '约为']
        calc_bonus = cls.TOKEN_BUDGET_HAS_CALCULATION if any(
            kw in (question or '') for kw in calc_keywords
        ) else 0

        total = base + length_bonus + doc_bonus + calc_bonus
        return max(cls.TOKEN_BUDGET_MIN, min(cls.TOKEN_BUDGET_MAX, total))

    # ================================================================
    # 检索配置
    # ================================================================
    RETRIEVAL_TOP_K = 20              # ① BM25 检索返回数(10→20, 提高证据命中率)
    EVIDENCE_MAX_PER_DOC = 8          # 每文档最多取证据片段数
    EVIDENCE_MAX_TOTAL = 16           # 总共最多证据片段数(翻倍)
    TABLE_WEIGHT_BOOST = 2.0          # ⑤ 表格内容权重加成

    # ================================================================
    # 缓存配置
    # ================================================================
    CACHE_MAX_SIZE = 100              # 缓存最大条目数

    # ================================================================
    # 内部方法
    # ================================================================
    @staticmethod
    def _load_dotenv(env_path: str = None) -> None:
        """手动加载 .env 文件（无需 python-dotenv）"""
        if env_path is None:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

        if not os.path.isfile(env_path):
            return

        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                # 去掉引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value

    def get_raw_path(self, domain: str) -> str:
        """获取某个领域的 raw 目录路径"""
        return os.path.join(self.raw_dir, domain)

    def get_question_path(self, group: str = "group_a", domain: str = None) -> str:
        """获取题目文件路径"""
        if domain:
            return os.path.join(self.questions_dir, group, f"{domain}_questions.json")
        return os.path.join(self.questions_dir, group)


# 单例 —— 整个项目共用这一个配置实例
config = Config()
