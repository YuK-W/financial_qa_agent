# multi_step_reasoning.py
class MultiStepReasoner:
    def __init__(self, agent):
        self.agent = agent
    
    def reason_with_verification(self, question, options, evidences):
        """多轮推理+自我验证"""
        
        # 第一轮：初步推理
        preliminary = self.initial_reasoning(question, options, evidences)
        
        # 第二轮：证据回溯验证
        verified = self.verify_with_evidence(preliminary, evidences)
        
        # 第三轮：一致性检查
        consistent = self.check_consistency(verified, options)
        
        return consistent
    
    def initial_reasoning(self, question, options, evidences):
        """初步推理"""
        prompt = f"""
        请分析以下问题，找出每个选项的关键判断点。
        
        问题：{question}
        
        选项：
        A: {options['A']}
        B: {options['B']}
        C: {options['C']}
        D: {options['D']}
        
        请输出每个选项需要的证据类型：
        """
        return self.agent.call_qwen(prompt)
    
    def verify_with_evidence(self, reasoning, evidences):
        """用证据验证推理结果"""
        # 为每个选项提取相关证据
        verified_result = {}
        # 实现验证逻辑
        return verified_result