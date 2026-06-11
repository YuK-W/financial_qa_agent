class DomainPromptBuilder:
    @staticmethod
    def insurance_prompt(question, options, evidences):
        """保险领域专用提示词"""
        return f"""
        你是一个保险条款分析专家。请严格依据以下保险条款内容回答问题。
        
        ## 保险条款摘要
        {evidences}
        
        ## 问题
        {question}
        
        ## 选项
        A: {options['A']}
        B: {options['B']}
        C: {options['C']}
        D: {options['D']}
        
        ## 分析要点
        1. 仔细核对保险责任的范围和限制条件
        2. 注意除外责任条款
        3. 计算现金价值时注意缴费年限和退保时间
        4. 身故保险金注意区分意外身故和疾病身故
        
        请逐项判断每个选项的正确性，并给出理由。
        """
    
    @staticmethod
    def regulation_prompt(question, options, evidences):
        """监管法规专用提示词"""
        return f"""
        你是一个法律合规专家。请严格依据以下法律法规条文回答问题。
        
        ## 法规条文
        {evidences}
        
        ## 问题
        {question}
        
        ## 选项
        A: {options['A']}
        B: {options['B']}
        C: {options['C']}
        D: {options['D']}
        
        ## 分析要点
        1. 严格匹配法条原文，不要自行解释
        2. 注意"应当"、"可以"、"必须"等强制性词汇
        3. 注意适用条件和例外情形
        4. 注意法规的施行日期和适用范围
        
        请逐项判断每个选项的正确性。
        """