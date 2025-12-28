import re
import jieba
import jieba.posseg as pseg
from typing import List, Tuple, Optional
from collections import defaultdict

class SemanticTextSplitter:
    """
    优化版轻量级中文语义文本分割器
    重点解决句号归属问题，确保句号与前面的句子内容在一起
    优化换行符处理，换行符具有最高优先级，出现换行符强制断句
    增加连字符预防处理，避免在后续英文处理中被误删
    使用jieba进行中文分词和词性标注，结合语法规则进行智能句子分割
    执行速度快，适合中文文本处理
    """
    
    def __init__(self):
        # 初始化jieba，加载词典
        jieba.initialize()
        
        # 定义语义分割的关键标点符号
        self.sentence_endings = ['。', '！', '？', '…', '；', '!', '?', ';', '．', '！', '？']
        self.clause_separators = ['，', '、', '：', ',', ':', '——', '－', '、']
        
        # 语义连贯性关键词
        self.coherence_words = {
            'conjunction': ['但是', '然而', '因此', '所以', '因为', '虽然', '即使', '如果', '当', '在', '从', '到', '和', '与', '及', '或'],
            'pronoun': ['他', '她', '它', '这', '那', '此', '其', '这些', '那些', '这个', '那个'],
            'continuation': ['然后', '接着', '随后', '同时', '此外', '另外', '而且', '并且']
        }
        
        # 连字符处理映射
        self.hyphen_replacement_map = {
            '-': '—',  # 将英文连字符替换为中文破折号（不会被删除）
            '--': '——', # 双连字符替换为中文长破折号
            '—': '—',  # 保留已有的中文破折号
            '－': '—'   # 全角连字符替换为中文破折号
        }
        
        # 用于还原的映射（如果需要）
        self.replacement_to_original = {
            '—': '-',   # 中文破折号还原为英文连字符
            '——': '--'  # 中文长破折号还原为双连字符
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        预处理文本，特别处理连字符问题
        将中文语境中的连字符替换为不会被删除的符号
        """
        if not text:
            return text
        
        # 1. 先处理连续的连字符
        text = re.sub(r'---+', '——', text)  # 三个以上连字符替换为长破折号
        text = re.sub(r'--', '——', text)    # 双连字符替换为长破折号
        
        # 2. 处理单个连字符，需要判断是否在中文语境中
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            
            if char == '-':
                # 检查上下文是否为中文语境
                is_chinese_context = False
                
                # 检查前一个字符
                if i > 0:
                    prev_char = text[i-1]
                    if '\u4e00' <= prev_char <= '\u9fff' or prev_char in  ["'","，","。","、","；","：",'"',"【","】","《","》"," "]:
                        is_chinese_context = True
                
                # 检查后一个字符
                if i < len(text) - 1:
                    next_char = text[i+1]
                    if '\u4e00' <= next_char <= '\u9fff' or next_char in  ["'","，","。","、","；","：",'"',"【","】","《","》"," "]:
                        is_chinese_context = True
                
                # 检查是否在数字之间（表示范围）
                if i > 0 and i < len(text) - 1:
                    prev_char = text[i-1]
                    next_char = text[i+1]
                    if prev_char.isdigit() and next_char.isdigit():
                        is_chinese_context = True  # 数字范围在中文中也需要保留
                
                # 根据上下文决定替换方式
                if is_chinese_context:
                    result.append('—')  # 替换为中文破折号
                else:
                    result.append('-')  # 保留英文连字符
                i += 1
            else:
                result.append(char)
                i += 1
        
        processed_text = ''.join(result)
        
        # 3. 处理其他需要转义的符号
        processed_text = self._handle_special_characters(processed_text)
        
        return processed_text
    
    def _handle_special_characters(self, text: str) -> str:
        """
        处理其他特殊字符，确保它们在后续处理中不会丢失语义
        """
        # 将英文引号替换为中文引号（如果在中文语境中）
        text = re.sub(r'(?<=[\u4e00-\u9fff])"(?=[\u4e00-\u9fff])', '“', text)
        text = re.sub(r'(?<=[\u4e00-\u9fff])"(?=[^，。、；：！？])', '”', text)
        
        # 将英文括号在中文语境中替换为中文括号
        text = re.sub(r'(?<=[\u4e00-\u9fff])\((?=[\u4e00-\u9fff])', '（', text)
        text = re.sub(r'(?<=[\u4e00-\u9fff])\)(?=[\u4e00-\u9fff])', '）', text)
        
        # 修复可能的标点符号问题
        text = re.sub(r'([。！？…；])\s+', r'\1', text)  # 移除句号后的多余空格
        text = re.sub(r'\s+([。！？…；])', r'\1', text)  # 移除句号前的多余空格
        
        return text
    
    def postprocess_segments(self, segments: List[str]) -> List[str]:
        """
        后处理分割结果，清理标点符号问题
        """
        result = []
        for segment in segments:
            # 清理标点符号前后的空格
            segment = re.sub(r'\s+([。！？…；，、：])', r'\1', segment)  # 移除标点前的空格
            segment = re.sub(r'([。！？…；，、：])\s+', r'\1', segment)  # 移除标点后的空格
            
            # 确保句号等结束标点不会单独成段
            segment = segment.strip()
            if segment and not re.match(r'^[。！？…；,.;?!]$', segment):
                result.append(segment)
        
        return result
    
    def gyc_split_text(self, text: str, min_text_len: int = 30, max_text_len: int = 60, 
                      preserve_hyphens: bool = True) -> List[str]:
        """
        基于语义的中文文本分割
        优先考虑语义完整性，同时满足长度约束
        换行符具有最高优先级，强制断句
        preserve_hyphens: 是否保留连字符处理（True时进行预防性替换）
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # 预处理：处理连字符和其他特殊字符
        processed_text = text.strip()
        if preserve_hyphens:
            processed_text = self.preprocess_text(processed_text)
        
        # 1. 首先按换行符进行强制分割（最高优先级）
        line_segments = self._split_by_newlines(processed_text)
        
        # 2. 对每个换行片段进行进一步语义分割
        result_segments = []
        for segment in line_segments:
            if not segment.strip():
                continue
            
            # 短片段直接保留
            if len(segment) <= max_text_len:
                result_segments.append(segment.strip())
                continue
            
            # 长片段进行智能分割
            sub_segments = self._semantic_segment(segment, min_text_len, max_text_len)
            result_segments.extend(sub_segments)
        
        # 3. 优化片段长度
        optimized_segments = self._optimize_segment_lengths(result_segments, min_text_len, max_text_len)
        
        # 4. 后处理：清理标点符号问题
        optimized_segments = self.postprocess_segments(optimized_segments)
        
        # 5. 二次优化：确保没有孤立的标点符号
        optimized_segments = self._fix_isolated_punctuation(optimized_segments)
        
        return optimized_segments
    
    def _split_by_newlines(self, text: str) -> List[str]:
        """
        按换行符进行强制分割，换行符具有最高优先级
        修复句号归属问题：确保换行符不会将句号分割到下一行
        """
        # 替换Windows换行符为统一格式
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n+', '\n', text)  # 将连续换行符压缩为单个
        
        # 修复常见的句号分割问题
        text = re.sub(r'([。！？…；])\n', r'\1', text)  # 句号后换行 -> 句号
        text = re.sub(r'\n([。！？…；])', r'\1', text)  # 换行后句号 -> 句号
        
        # 按单个换行符分割
        lines = text.split('\n')
        
        # 清理空白
        result = []
        for line in lines:
            line = line.strip()
            if line:  # 保留非空行
                result.append(line)
        
        return result
    
    def _semantic_segment(self, text: str, min_len: int, max_len: int) -> List[str]:
        """
        基于语义的智能分割
        重点修复句号归属问题：确保句号总是与前面的句子内容在一起
        """
        if len(text) <= max_len:
            return [self._fix_punctuation_at_end(text.strip())]
        
        segments = []
        current_segment = ""
        current_length = 0
        
        # 使用jieba进行词性标注
        words = list(pseg.cut(text))
        
        i = 0
        while i < len(words):
            word, flag = words[i]
            word_len = len(word)
            
            # 重点修复：句号归属问题
            if word in self.sentence_endings:
                # 将句号添加到当前片段，然后分割
                current_segment += word
                current_length += word_len
                
                # 确保当前片段不为空且长度合理
                if current_segment.strip() and len(current_segment.strip()) >= 3:
                    segments.append(current_segment.strip())
                    current_segment = ""
                    current_length = 0
                i += 1
                continue
            
            # 检查是否应该分割
            should_split = self._should_split_here(word, flag, current_length, max_len)
            
            if should_split and current_segment.strip():
                # 分割前确保清理标点符号
                current_segment = self._fix_punctuation_at_end(current_segment)
                segments.append(current_segment.strip())
                current_segment = ""
                current_length = 0
            
            current_segment += word
            current_length += word_len
            i += 1
        
        # 添加最后一个片段
        if current_segment.strip():
            current_segment = self._fix_punctuation_at_end(current_segment)
            segments.append(current_segment.strip())
        
        # 合并可能出现的孤立标点
        return self._merge_isolated_punctuation(segments)
    
    def _should_split_here(self, word: str, pos_tag: str, current_len: int, max_len: int) -> bool:
        """
        判断是否应该在当前位置分割
        修复句号归属问题，避免在标点符号前分割
        """
        # 规则0：如果是标点符号，不要在这里分割（让标点符号归属到前面的内容）
        if word in self.sentence_endings + self.clause_separators + ['—', '——']:
            return False
        
        # 规则1：长度超过限制 - 但在标点符号处分割
        if current_len + len(word) > max_len:
            # 寻找最近的标点符号作为分割点
            return True
        
        # 规则2：遇到子句分隔符且长度接近max_len
        # 注意：这个检查已经移动到 _semantic_segment 中，避免在标点处分割
        
        # 规则3：语义连贯性检查 - 避免在连接词前分割
        if word in self.coherence_words['conjunction']:
            if current_len < max_len * 0.8:  # 更宽松的条件
                return False
        
        # 规则4：代词检查 - 避免在代词前分割
        if word in self.coherence_words['pronoun']:
            if current_len < max_len * 0.7:
                return False
        
        # 规则5：在自然停顿点前分割，但不包括标点符号
        if current_len > max_len * 0.7:
            return True
        
        return False
    
    def _fix_punctuation_at_end(self, text: str) -> str:
        """
        修复文本末尾的标点符号问题
        确保标点符号正确归属，不会出现孤立的标点
        """
        if not text:
            return text
        
        text = text.strip()
        
        # 如果文本以空格开头，移除
        text = re.sub(r'^\s+', '', text)
        
        # 如果文本以多个标点结尾，只保留最后一个
        text = re.sub(r'([。！？…；，、：])\1+', r'\1', text)
        
        # 如果文本以子句分隔符结尾，且长度很短，尝试合并
        if re.match(r'^[，、：,$:;]', text) and len(text) < 5:
            return text
        
        # 确保句号等结束标点在末尾
        if text and text[-1] in ['，', '、', ':', ',', ';']:
            # 将逗号替换为句号
            text = text[:-1] + '。'
        
        return text.strip()
    
    def _merge_isolated_punctuation(self, segments: List[str]) -> List[str]:
        """
        合并孤立的标点符号片段
        例如：['第一句。', '。', '第二句'] -> ['第一句。', '第二句']
        """
        result = []
        i = 0
        
        while i < len(segments):
            current = segments[i].strip()
            
            # 检查是否是孤立的标点符号
            if re.match(r'^[。！？…；，、：,!?:;]$', current):
                # 尝试合并到前一个片段
                if result:
                    last_segment = result[-1]
                    if last_segment and last_segment[-1] not in self.sentence_endings:
                        result[-1] = last_segment + current
                    # 如果前一个片段已有结束标点，跳过当前孤立标点
                i += 1
                continue
            
            # 检查是否以标点符号开头
            if current and current[0] in ['。', '！', '？', '…', '；', ',', ';', ':']:
                # 尝试合并到前一个片段
                if result:
                    last_segment = result[-1]
                    if last_segment and last_segment[-1] not in self.sentence_endings:
                        result[-1] = last_segment + current
                        i += 1
                        continue
            
            result.append(current)
            i += 1
        
        return result
    
    def _optimize_segment_lengths(self, segments: List[str], min_len: int, max_len: int) -> List[str]:
        """
        优化片段长度，合并过短片段，分割过长片段
        修复句号归属问题
        """
        result = []
        i = 0
        
        while i < len(segments):
            current = segments[i].strip()
            current_len = len(current)
            
            # 情况1：片段长度合适，直接添加
            if min_len <= current_len <= max_len:
                result.append(current)
                i += 1
                continue
            
            # 情况2：片段过短，尝试合并
            if current_len < min_len:
                # 2.1 尝试与前一个片段合并
                if result:
                    last_segment = result[-1]
                    # 检查是否可以语义合并
                    if (not last_segment.endswith(tuple(self.sentence_endings)) and 
                        len(last_segment + current) <= max_len * 1.3):
                        result[-1] = last_segment + current
                        i += 1
                        continue
                
                # 2.2 尝试与后一个片段合并
                if i + 1 < len(segments):
                    next_segment = segments[i + 1].strip()
                    if (not current.endswith(tuple(self.sentence_endings)) and 
                        len(current + next_segment) <= max_len * 1.3):
                        result.append(current + next_segment)
                        i += 2
                        continue
                
                # 2.3 无法合并，单独保留
                result.append(current)
                i += 1
                continue
            
            # 情况3：片段过长，需要分割
            if current_len > max_len:
                # 智能分割长文本，确保句号归属
                sub_segments = self._split_long_text_with_punctuation_fix(current, min_len, max_len)
                result.extend(sub_segments)
                i += 1
                continue
            
            # 默认情况
            result.append(current)
            i += 1
        
        return [seg.strip() for seg in result if seg.strip()]
    
    def _split_long_text_with_punctuation_fix(self, text: str, min_len: int, max_len: int) -> List[str]:
        """
        智能分割长文本，确保句号等标点符号正确归属
        """
        # 优先按句子结束符分割
        sentence_endings = self.sentence_endings
        
        # 找到所有句子结束位置
        split_points = []
        for i, char in enumerate(text):
            if char in sentence_endings:
                split_points.append(i + 1)  # 包括标点符号
        
        if not split_points:
            # 没有句子结束符，按子句分隔符分割
            return self._split_by_clause_separators(text, min_len, max_len)
        
        segments = []
        start = 0
        
        for end in split_points:
            segment = text[start:end].strip()
            if len(segment) > max_len:
                # 子片段仍然过长，进一步分割
                sub_segments = self._split_by_clause_separators(segment, min_len, max_len)
                segments.extend(sub_segments)
            else:
                segments.append(segment)
            start = end
        
        # 处理剩余部分
        if start < len(text):
            remaining = text[start:].strip()
            if remaining:
                segments.append(remaining)
        
        return segments
    
    def _split_by_clause_separators(self, text: str, min_len: int, max_len: int) -> List[str]:
        """
        按子句分隔符分割，确保标点符号归属
        """
        clause_separators = self.clause_separators + ['—', '——']
        
        # 找到所有子句分隔符位置
        split_points = []
        for i, char in enumerate(text):
            if char in clause_separators:
                split_points.append(i + 1)  # 包括标点符号
        
        if not split_points:
            # 没有子句分隔符，按长度分割
            return [text[i:i+max_len] for i in range(0, len(text), max_len)]
        
        segments = []
        start = 0
        
        for end in split_points:
            segment = text[start:end].strip()
            segments.append(segment)
            start = end
        
        if start < len(text):
            segments.append(text[start:].strip())
        
        return segments
    
    def _fix_isolated_punctuation(self, segments: List[str]) -> List[str]:
        """
        二次修复：确保没有孤立的标点符号
        """
        result = []
        for segment in segments:
            segment = segment.strip()
            
            # 跳过孤立的标点符号
            if re.match(r'^[。！？…；，、：,!?:;.，；：、]$', segment):
                continue
            
            # 修复以标点符号开头的片段
            if segment and segment[0] in ['。', '！', '？', '…', '；', ',', ';', ':']:
                if result:
                    result[-1] = result[-1] + segment[0]
                    segment = segment[1:].strip()
            
            # 修复以标点符号结尾的短片段
            if len(segment) <= 3 and segment and segment[-1] in ['，', '、', ':', ',', ';']:
                if result:
                    result[-1] = result[-1] + segment
                    continue
            
            if segment:
                result.append(segment)
        
        return result
    
    def _check_semantic_coherence(self, seg1: str, seg2: str) -> bool:
        """
        检查两个片段是否具有语义连贯性，适合合并
        """
        if not seg1 or not seg2:
            return False
        
        # 检查seg1是否以连接词结尾
        if seg1.endswith(('，', '、', '：', ',', ':', '，', '—', '——')):
            return True
        
        # 检查seg2是否以连接词开头
        connective_starts = ['但是', '然而', '因此', '所以', '因为', '虽然', '即使', '如果', '当', '在', '从', '到', '和', '与', '及', '或', '他', '她', '它', '这', '那', '此', '其', '—', '——']
        for word in connective_starts:
            if seg2.startswith(word):
                return True
        
        # 检查是否构成完整语义
        if (seg1 and seg1[-1] in ['，', '、', '：', ',', ':', '—', '——'] and 
            seg2 and seg2[0] not in self.sentence_endings):
            return True
        
        return False

# 使用示例
if __name__ == "__main__":
    splitter = SemanticTextSplitter()
    
    # 测试文本 - 重点测试句号归属问题
    test_text = """
    第一句。第二句，包含逗号。第三句很长，需要分割，但是句号不能单独出来。
    第四句。第五句。
    第六句包含错误的句号分割.
    第七句，标点符号问题，，，：：
    第八句，最后一个词是逗号，
    第九句。孤立句号。
    第十句正常结束。
    
    第十一句在新段落。
    """
    
    print("=== 原始文本 ===")
    print(test_text)
    
    print("\n=== 修复句号归属后的分割结果 ===")
    segments = splitter.gyc_split_text(test_text, min_text_len=15, max_text_len=40)
    
    for i, seg in enumerate(segments):
        print(f"片段 {i+1} (长度: {len(seg)}): {seg}")
    
    # 额外测试：极端句号问题
    print("\n\n=== 极端句号测试 ===")
    extreme_text = "。孤立句号测试。正常句子。另一个孤立句号.。最后一个句子。"
    segments = splitter.gyc_split_text(extreme_text, min_text_len=5, max_text_len=30)
    
    for i, seg in enumerate(segments):
        print(f"片段 {i+1} (长度: {len(seg)}): '{seg}'")
    
    # 复杂混合测试
    print("\n\n=== 复杂中英文混合测试 ===")
    mixed_text = """
    AI发展-state-of-the-art：人工智能技术日新月异。2024-2025年规划。
    价格范围：100-500元。时间安排：周一-周五。
    中文处理-text-processing非常重要。machine-learning-deep-learning是热门领域。
    第一句。第二句。第三句。
    第四句，包含很多标点符号，，，：：：！！！
    """
    segments = splitter.gyc_split_text(mixed_text, min_text_len=10, max_text_len=35)
    
    for i, seg in enumerate(segments):
        print(f"片段 {i+1} (长度: {len(seg)}): {seg}")