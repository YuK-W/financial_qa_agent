# domain_prompts.py
"""
领域专用Prompt模板 — 5领域完整覆盖

每个模板包含: 角色设定 + 分析框架 + 输出约束
解决通用Prompt对金融术语识别率低、推理方向偏差的问题。
"""


class DomainPromptBuilder:
    """按领域构建专用Prompt，替代Agent内嵌的通用模板"""

    # ================================================================
    # 5 个领域模板（完整覆盖）
    # ================================================================

    @staticmethod
    def financial_contracts_prompt(question: str, options: dict, context: str) -> str:
        """金融合同 — 债券募集说明书、贷款合同、担保合同"""
        options_text = DomainPromptBuilder._fmt_options(options)
        return (
            "你是一位资深债券发行审核专家，精通募集说明书、承销协议、担保合同的条款分析。\n\n"
            f"## 文档证据\n{context}\n\n"
            f"## 问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            "## 分析框架\n"
            "1. 发行人身份: 核对发行人全称、注册地址、法定代表人是否与文档一致\n"
            "2. 发行要素: 发行规模、债券期限、票面利率、信用评级是否与募集说明书匹配\n"
            "3. 中介机构: 主承销商、受托管理人、评级机构、律师事务所、会计师事务所名称是否准确\n"
            "4. 担保条款: 担保方式(连带责任保证/抵押/质押)、担保范围、担保期限\n"
            "5. 特殊条款: 回售、赎回、调整票面利率、交叉违约等是否在文档中存在\n\n"
            "## 推理步骤(Chain-of-Thought)\n"
            "1. 从证据中提取关键事实(逐条列出，引用原文)\n"
            "2. 对A/B/C/D每个选项独立判断: 将选项与事实对比→[正确/错误]→引用证据原文\n"
            "   重要: 每个选项独立判断，可以有多个正确，也可能全错\n"
            "3. 汇总正确选项，以'正确答案：X'输出"
        )

    @staticmethod
    def financial_reports_prompt(question: str, options: dict, context: str) -> str:
        """财务报告 — 年报、审计报告"""
        options_text = DomainPromptBuilder._fmt_options(options)
        return (
            "你是一位注册会计师，精通上市公司年度报告和审计报告的财务数据分析。\n\n"
            f"## 文档证据\n{context}\n\n"
            f"## 问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            "## 分析框架\n"
            "1. 营收与利润: 营业收入、营业成本、净利润的绝对值及同比变化\n"
            "2. 资产负债: 总资产、总负债、净资产、资产负债率的精确数值\n"
            "3. 现金流量: 经营/投资/筹资活动现金流净额及变化方向\n"
            "4. 会计政策: 收入确认方法、折旧方法、合并报表范围是否变更\n"
            "5. 审计意见: 标准无保留/保留/否定/无法表示意见类型\n\n"
            "## 推理步骤(Chain-of-Thought)\n"
            "1. 从证据中提取关键财务数据(逐条列出，带单位)\n"
            "2. 统一单位后，对A/B/C/D每个选项独立判断: 将选项数值与证据数值对比→[正确/错误]→引用证据数值\n"
            "   重要: 每个选项独立判断，注意区分亿元/万元/元\n"
            "3. 汇总正确选项，以'正确答案：X'输出"
        )

    @staticmethod
    def insurance_prompt(question: str, options: dict, context: str) -> str:
        """保险条款"""
        options_text = DomainPromptBuilder._fmt_options(options)
        return (
            "你是一位资深保险精算师，精通人身保险/财产保险条款的保险责任与免责条款分析。\n\n"
            f"## 文档证据\n{context}\n\n"
            f"## 问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            "## 分析框架\n"
            "1. 保险责任: 保障范围、赔付条件、等待期的精确约定\n"
            "2. 责任免除: 哪些情形不在保障范围内\n"
            "3. 保险金额与保费: 保额计算方式、费率表、缴费期间\n"
            "4. 现金价值: 退保时的现金价值计算(缴费年限×退保时间)\n"
            "5. 受益人: 指定受益人与法定受益人的区别\n"
            "6. 特殊约定: 意外身故vs疾病身故的赔付差异、免赔额、赔付比例\n\n"
            "## 推理步骤(Chain-of-Thought)\n"
            "1. 从条款中提取关键约定(逐条列出，引用条款原文)\n"
            "2. 对A/B/C/D每个选项独立判断: 与条款约定对比→[正确/错误]→引用条款证据\n"
            "   重要: 区分意外身故/疾病身故赔付条件，注意免责条款\n"
            "3. 汇总正确选项，以'正确答案：X'输出"
        )

    @staticmethod
    def regulatory_prompt(question: str, options: dict, context: str) -> str:
        """监管法规"""
        options_text = DomainPromptBuilder._fmt_options(options)
        return (
            "你是一位金融合规律师，精通中国金融监管法规(银保监/证监会/央行)的条文解释。\n\n"
            f"## 文档证据\n{context}\n\n"
            f"## 问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            "## 分析框架\n"
            "1. 法条定位: 条款编号、章节归属、法规名称是否与引用一致\n"
            "2. 强制程度: '应当'(必须)/'可以'(选择)/'不得'(禁止)的法律效力区分\n"
            "3. 适用条件: 主体资格、数额门槛、时间窗口等触发条件\n"
            "4. 例外情形: 是否存在但书条款或豁免条件\n"
            "5. 罚则: 违规后果(罚款/暂停业务/吊销执照)与法条是否对应\n"
            "6. 时效: 法规施行日期、过渡期安排是否影响适用\n\n"
            "## 推理步骤(Chain-of-Thought)\n"
            "1. 从法条中提取关键条文(逐条列出，引用法条原文+条款号)\n"
            "2. 对A/B/C/D每个选项独立判断: 与法条原文对比→[正确/错误]→引用法条证据\n"
            "   重要: 区分'应当'(必须)/'可以'(选择)/'不得'(禁止)的法律效力\n"
            "3. 汇总正确选项，以'正确答案：X'输出"
        )

    @staticmethod
    def research_prompt(question: str, options: dict, context: str) -> str:
        """研究报告 — 券商研报、行业分析"""
        options_text = DomainPromptBuilder._fmt_options(options)
        return (
            "你是一位证券研究所首席分析师，精通行业研究报告和公司深度报告的解读。\n\n"
            f"## 文档证据\n{context}\n\n"
            f"## 问题\n{question}\n\n"
            f"## 选项\n{options_text}\n\n"
            "## 分析框架\n"
            "1. 投资评级: 买入/增持/中性/减持/卖出的具体评级及变动\n"
            "2. 目标价: 目标价与当前价、估值方法(PE/PB/DCF)是否与报告一致\n"
            "3. 盈利预测: 营收增速、净利润增速、EPS预测值\n"
            "4. 行业地位: 市场份额、竞争优势、行业排名\n"
            "5. 风险提示: 政策风险/市场风险/经营风险的具体表述\n\n"
            "## 推理步骤(Chain-of-Thought)\n"
            "1. 从研报中提取关键结论与数据(逐条列出，引用原文)\n"
            "2. 对A/B/C/D每个选项独立判断: 与研报原文对比→[正确/错误]→引用证据\n"
            "   重要: 区分'事实陈述'与'分析师预测'('预计'/'可能'/'有望'是预测非事实)\n"
            "3. 汇总正确选项，以'正确答案：X'输出"
        )

    # ================================================================
    # 通用构建方法
    # ================================================================

    # 领域→模板方法映射
    _TEMPLATES = {
        'financial_contracts': financial_contracts_prompt,
        'financial_reports': financial_reports_prompt,
        'insurance': insurance_prompt,
        'regulatory': regulatory_prompt,
        'research': research_prompt,
    }

    @classmethod
    def build_prompt(cls, domain: str, question: str, options: dict, context: str) -> str:
        """
        根据领域自动选择对应的Prompt模板。

        Args:
            domain: 领域名 ('financial_contracts', 'financial_reports', ...)
            question: 问题文本
            options: 选项字典 {'A': '...', 'B': '...', ...}
            context: 检索到的文档证据文本

        Returns:
            完整的领域专用Prompt字符串
        """
        template_fn = cls._TEMPLATES.get(domain)
        if template_fn is None:
            # 未知领域，回退到通用金融合同模板
            template_fn = cls.financial_contracts_prompt
        return template_fn(question, options, context)

    @staticmethod
    def _fmt_options(options: dict) -> str:
        """格式化选项文本"""
        parts = []
        for key in ['A', 'B', 'C', 'D']:
            if key in options:
                parts.append(f"{key}. {options[key]}")
        return '\n'.join(parts)


# ================================================================
# 测试
# ================================================================
if __name__ == '__main__':
    builder = DomainPromptBuilder
    test_opts = {'A': '选项A内容', 'B': '选项B内容', 'C': '选项C内容', 'D': '选项D内容'}

    for domain in ['financial_contracts', 'financial_reports', 'insurance',
                    'regulatory', 'research']:
        prompt = builder.build_prompt(domain, '测试问题?', test_opts, '测试上下文')
        print(f"[{domain}] Prompt长度: {len(prompt)} 字符")
        print(f"  含'分析框架': {'分析框架' in prompt}")
        print(f"  含'正确答案': {'正确答案' in prompt}")

    print("\nP1-1: 5领域模板全部就绪")
