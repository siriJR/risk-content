import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import umap
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import jieba
import re
from typing import List, Dict, Tuple, Any
import warnings
import jieba.analyse
from src.TextPreprocess import TextPreprocess

from tqdm import tqdm

tqdm.pandas()
tl = TextPreprocess()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')


class ChatTopicAnalyzer:
    def __init__(self, model_name='BAAI/bge-base-zh'):
        """
        初始化主题分析器
        Args:
            model_name: 使用的sentence transformer模型
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.cluster_labels = None
        self.df = None
        self.valid_indices = None  # 新增：保存有效文本的索引

        # 扩展停用词列表
        self.extended_stop_words = self._get_extended_stop_words()

        # 无意义词模式
        self.meaningless_patterns = [
            r'^[嘛呢吧啊呀哦嗯哈呵唉喔哇]$',
            r'^[是不是有没有能不能会不会]$',
            r'^[这个那个这些那些这样那样]$',
            r'^[然后所以因为但是不过可是]$',
            r'^[哈哈呵呵嘿嘿嘻嘻哈哈]$'
        ]

        # 词性白名单（只保留这些词性的词）
        self.allowed_pos = {'n', 'v', 'a', 'vn', 'an', 'nr', 'ns', 'nt', 'nz'}

    def _get_extended_stop_words(self) -> List[str]:
        """
        获取扩展的中文停用词列表
        Returns:
            停用词列表
        """
        common_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说',
            '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们', '你们', '他们',
            '啊', '哦', '嗯', '呀', '哈', '哈哈', '呵呵', '哦哦', '嗯嗯', '啊啊', '哇', '哇塞', '啥', '什么', '怎么', '为什么',
            '如何', '哪个', '哪里', '谁', '吗', '呢', '吧', '啊', '啦', '诶', '哎', '哼', '呃', '嘘', '哇', '喔', '喂', '嘛',
            '噢', '呦', '咳', '呸', '嘻嘻', '嘿嘿', '哼哧', '呜', '呜呼', '这个', '那个', '这些', '那些', '这样', '那样',
            '这么', '那么', '这样', '那样', '可以', '应该', '能够', '可能', '一定', '必须', '需要', '想要', '希望', '觉得',
            '认为', '知道', '明白', '理解', '认识', '看到', '听到', '感到', '想到', '做到', '完成', '开始', '结束', '继续',
            '就是', '可是', '但是', '不过', '然后', '所以', '因为', '如果', '的话', '一下', '一点', '一些', '一种', '一样',
            '一般', '一直', '一起', '一下', '一点', '不会', '不能', '不要', '不用', '不行', '不好', '不错', '不对', '不过',
            '不会', '不能', '不要', '不用', '不行', '不好', '不错', '不对', '不过', '不会', '不能', '不要', '不用', '不行'
        }
        return list(common_stop_words)

    def _is_meaningful_word(self, word: str) -> bool:
        """
        判断词语是否有实际意义
        """
        # 基础过滤
        if (len(word) <= 1 or
                word in self.extended_stop_words or
                word.isdigit() or
                word.encode('utf-8').isalpha()):
            return False

        # 模式匹配过滤
        for pattern in self.meaningless_patterns:
            if re.match(pattern, word):
                return False

        return True

    def _segment_and_filter(self, texts: List[str]) -> List[List[str]]:
        """
        统一的分词和过滤处理
        """
        segmented_results = []
        for text in texts:
            # 使用jieba分词并获取词性
            words_with_pos = jieba.posseg.cut(text)
            filtered_words = []

            for word, pos in words_with_pos:
                # 词性过滤 + 基础过滤
                if (pos[0] in self.allowed_pos and
                        self._is_meaningful_word(word)):
                    filtered_words.append(word)

            segmented_results.append(filtered_words)

        return segmented_results

    def method_tfidf_optimized(self, texts: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        方法1：优化的TF-IDF
        """
        if len(texts) < 3:
            return []

        # 分词和过滤
        segmented_texts = self._segment_and_filter(texts)
        text_strings = [' '.join(words) for words in segmented_texts]

        try:
            # 更严格的TF-IDF参数
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=self.extended_stop_words,
                min_df=1,  # 至少在2个文档中出现
                max_df=0.7,  # 最多在70%的文档中出现
                ngram_range=(1, 2)
            )

            X = vectorizer.fit_transform(text_strings)
            words = vectorizer.get_feature_names_out()
            scores = np.array(X.sum(axis=0)).flatten()

            # 组合结果
            results = list(zip(words, scores))
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:top_k]

        except Exception as e:
            print(f"TF-IDF提取错误: {e}")
            return []

    def method_textrank(self, texts: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        方法2：TextRank
        """
        if len(texts) < 3:
            return []

        # 合并文本
        combined_text = ' '.join([' '.join(self._segment_and_filter([text])[0]) for text in texts])

        if len(combined_text.strip()) < 10:
            return []

        try:
            # 使用jieba的TextRank实现
            keywords = jieba.analyse.textrank(
                combined_text,
                topK=top_k * 3,  # 多取一些用于过滤
                withWeight=True,
                allowPOS=('n', 'vn', 'v', 'a')  # 名词、动名词、动词、形容词
            )

            # 过滤无意义词
            filtered_keywords = []
            for word, score in keywords:
                if self._is_meaningful_word(word):
                    filtered_keywords.append((word, score))
                if len(filtered_keywords) >= top_k:
                    break

            return filtered_keywords

        except Exception as e:
            print(f"TextRank提取错误: {e}")
            return []

    def method_semantic(self, texts: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        方法3：基于语义的方法
        """
        if len(texts) < 3:
            return []

        # 生成文本嵌入（如果还没有）
        if self.embeddings is None:
            processed_texts = [self.preprocess_text(text) for text in texts]
            valid_texts = [text for text in processed_texts if len(text) > 3]
            if len(valid_texts) < 3:
                return []
            self.embeddings = self.model.encode(valid_texts, show_progress_bar=True)

        # 提取候选词
        candidate_words = []
        word_occurrences = {}

        segmented_texts = self._segment_and_filter(texts)

        for i, words in enumerate(segmented_texts):
            for word in words:
                if word not in candidate_words:
                    candidate_words.append(word)
                if word not in word_occurrences:
                    word_occurrences[word] = []
                word_occurrences[word].append(i)

        # 过滤低频词
        candidate_words = [word for word in candidate_words if len(word_occurrences[word]) >= 2]

        if not candidate_words:
            return []

        # 计算词向量（通过包含该词的文本向量的平均）
        word_vectors = []
        valid_candidates = []

        for word in candidate_words:
            indices = word_occurrences[word]
            if len(indices) <= len(self.embeddings):
                word_vector = self.embeddings[indices].mean(axis=0)
                word_vectors.append(word_vector)
                valid_candidates.append(word)

        if not valid_candidates:
            return []

        word_vectors = np.array(word_vectors)

        # 计算与整体语义中心的相似度
        text_center = self.embeddings.mean(axis=0)
        similarities = cosine_similarity(word_vectors, [text_center]).flatten()

        # 组合结果
        results = list(zip(valid_candidates, similarities))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def method_hybrid(self, texts: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        方法4：混合方法（加权投票）
        """
        if len(texts) < 3:
            return []

        # 获取各种方法的结果
        tfidf_results = self.method_tfidf_optimized(texts, top_k * 3)
        textrank_results = self.method_textrank(texts, top_k * 3)
        semantic_results = self.method_semantic(texts, top_k * 3)

        # 投票得分系统
        keyword_scores = {}

        # TF-IDF得分（权重0.3）
        for i, (word, score) in enumerate(tfidf_results):
            normalized_score = 1.0 - (i / len(tfidf_results))  # 排名归一化
            keyword_scores[word] = keyword_scores.get(word, 0) + normalized_score * 0.3

        # TextRank得分（权重0.3）
        for i, (word, score) in enumerate(textrank_results):
            normalized_score = 1.0 - (i / len(textrank_results))
            keyword_scores[word] = keyword_scores.get(word, 0) + normalized_score * 0.3

        # 语义得分（权重0.4）
        for i, (word, score) in enumerate(semantic_results):
            normalized_score = 1.0 - (i / len(semantic_results))
            keyword_scores[word] = keyword_scores.get(word, 0) + normalized_score * 0.4

        # 按总分排序
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

        return [(word, score) for word, score in sorted_keywords[:top_k]]

    def extract_meaningful_keywords(self, texts: List[str], top_k: int = 8, method: str = 'hybrid') -> List[str]:
        """
        提取有意义的主题关键词 - 改进版本，支持多种方法
        Args:
            texts: 文本列表
            top_k: 返回前k个关键词
            method: 提取方法 ('tfidf', 'textrank', 'semantic', 'hybrid')
        Returns:
            关键词列表
        """
        if len(texts) < 3:
            return []

        try:
            if method == 'tfidf':
                keywords_with_scores = self.method_tfidf_optimized(texts, top_k)
            elif method == 'textrank':
                keywords_with_scores = self.method_textrank(texts, top_k)
            elif method == 'semantic':
                keywords_with_scores = self.method_semantic(texts, top_k)
            else:  # 默认使用混合方法
                keywords_with_scores = self.method_hybrid(texts, top_k)

            # 只返回关键词，不返回分数
            keywords = [word for word, score in keywords_with_scores]

            # 进一步过滤，确保关键词有意义
            meaningful_keywords = []
            for keyword in keywords:
                if (len(keyword) > 1 and
                        not keyword.isdigit() and
                        not all(ord(c) < 128 for c in keyword)):  # 排除纯英文
                    meaningful_keywords.append(keyword)

            return meaningful_keywords[:top_k]

        except Exception as e:
            print(f"关键词提取错误 ({method}): {e}")
            return []

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据
        Args:
            file_path: CSV文件路径
        Returns:
            DataFrame
        """
        self.df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"数据加载成功，共 {len(self.df)} 条记录")
        return self.df

    def preprocess_text(self, text: str) -> str:
        """
        文本预处理 - 增强版本
        Args:
            text: 原始文本
        Returns:
            处理后的文本
        """
        if pd.isna(text):
            return ""

        # 使用自定义预处理
        text = tl.preProcess(str(text).strip().lower())

        # 过滤过短的文本
        if len(text) < 2:
            return ""

        return text

    def generate_embeddings(self) -> np.ndarray:
        """
        生成文本嵌入向量
        Returns:
            嵌入向量数组
        """
        if self.df is None:
            raise ValueError("请先加载数据")

        # 预处理文本
        self.df['processed_msg'] = self.df['msg'].apply(self.preprocess_text)

        # 过滤空文本和过短文本
        valid_mask = (self.df['processed_msg'].str.len() > 3)  # 至少3个字符
        self.valid_indices = self.df[valid_mask].index  # 保存有效文本的索引
        valid_texts = self.df.loc[self.valid_indices, 'processed_msg'].tolist()

        print(f"生成嵌入向量，有效文本数量: {len(valid_texts)}")

        # 生成嵌入向量
        self.embeddings = self.model.encode(valid_texts, show_progress_bar=True)
        print(f"嵌入向量生成完成，维度: {self.embeddings.shape}")

        return self.embeddings

    def cluster_texts(self, n_clusters: int = None, method: str = 'kmeans') -> np.ndarray:
        """
        对文本进行聚类
        Args:
            n_clusters: 聚类数量，如果为None则自动确定
            method: 聚类方法 ('kmeans', 'dbscan')
        Returns:
            聚类标签
        """
        if self.embeddings is None:
            self.generate_embeddings()

        if method == 'kmeans':
            if n_clusters is None:
                # 使用肘部法则确定聚类数量
                n_clusters = self._find_optimal_clusters()
                print(f"自动确定的聚类数量: {n_clusters}")

            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = clusterer.fit_predict(self.embeddings)

        elif method == 'dbscan':
            # 使用UMAP降维后再进行DBSCAN聚类
            reducer = umap.UMAP(n_components=50, random_state=42)
            reduced_embeddings = reducer.fit_transform(self.embeddings)

            clusterer = DBSCAN(eps=0.5, min_samples=5)
            self.cluster_labels = clusterer.fit_predict(reduced_embeddings)

        # 将聚类标签添加到DataFrame中 - 修复形状不匹配问题
        self.df['cluster'] = -2  # 初始化为-2（无效文本）
        self.df.loc[self.valid_indices, 'cluster'] = self.cluster_labels

        print(f"聚类完成，共 {len(set(self.cluster_labels))} 个聚类")
        return self.cluster_labels

    def _find_optimal_clusters(self, max_k: int = 15) -> int:
        """
        使用肘部法则找到最佳聚类数量
        Args:
            max_k: 最大聚类数量
        Returns:
            最佳聚类数量
        """
        inertias = []
        k_range = range(2, min(max_k + 1, len(self.embeddings)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.embeddings)
            inertias.append(kmeans.inertia_)

        # 计算二阶导数找到拐点
        derivatives = np.diff(inertias)
        second_derivatives = np.diff(derivatives)
        optimal_k = k_range[np.argmin(second_derivatives) + 2]

        return optimal_k

    def generate_topic_labels(self, cluster_texts: List[str], keywords: List[str]) -> str:
        """
        为每个聚类生成有意义的主题标签
        Args:
            cluster_texts: 聚类中的文本
            keywords: 提取的关键词
        Returns:
            主题标签
        """
        if not cluster_texts or not keywords:
            return "未定义主题"

        # 分析文本内容特征
        text_samples = ' '.join(cluster_texts[:20])  # 取前20个样本进行分析

        # 基于关键词和文本内容生成主题标签
        if any(word in text_samples for word in ['唱歌', '声音', '听歌', '音乐', '好听']):
            return "音乐娱乐"
        elif any(word in text_samples for word in ['睡觉', '休息', '起床', '晚安', '熬夜']):
            return "作息生活"
        elif any(word in text_samples for word in ['吃饭', '餐厅', '美食', '火锅', '外卖']):
            return "饮食话题"
        elif any(word in text_samples for word in ['工作', '上班', '下班', '公司', '同事']):
            return "工作职场"
        elif any(word in text_samples for word in ['学习', '考试', '学校', '老师', '作业']):
            return "学习教育"
        elif any(word in text_samples for word in ['游戏', '打游戏', '玩家', '电竞', '王者']):
            return "游戏娱乐"
        elif any(word in text_samples for word in ['电影', '电视剧', '追剧', '影院', '综艺']):
            return "影视娱乐"
        elif any(word in text_samples for word in ['购物', '买', '淘宝', '京东', '拼多多']):
            return "购物消费"
        elif any(word in text_samples for word in ['运动', '健身', '跑步', '锻炼', '减肥']):
            return "运动健康"
        elif any(word in text_samples for word in ['天气', '气温', '下雨', '晴天', '温度']):
            return "天气话题"
        elif any(word in text_samples for word in ['感情', '恋爱', '喜欢', '爱', '分手']):
            return "情感交流"
        elif any(word in text_samples for word in ['朋友', '聚会', '见面', '约', '聊天']):
            return "社交活动"
        else:
            # 如果没有匹配到特定类别，使用前3个关键词生成标签
            return f"{'、'.join(keywords[:3])}相关"

    def analyze_topics(self, keyword_method: str = 'hybrid') -> Dict:
        """
        分析每个聚类的主题 - 改进版本
        Args:
            keyword_method: 关键词提取方法 ('tfidf', 'textrank', 'semantic', 'hybrid')
        Returns:
            主题分析结果
        """
        if self.cluster_labels is None:
            raise ValueError("请先进行聚类")

        topic_analysis = {}

        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # DBSCAN中的噪声点
                continue

            cluster_mask = self.df['cluster'] == cluster_id
            cluster_texts = self.df[cluster_mask]['processed_msg'].tolist()
            original_texts = self.df[cluster_mask]['msg'].tolist()

            if len(cluster_texts) < 5:  # 提高最小聚类规模要求
                continue

            # 使用指定的方法提取有意义的主题关键词
            keywords = self.extract_meaningful_keywords(cluster_texts, method=keyword_method)

            # 生成主题标签
            topic_label = self.generate_topic_labels(cluster_texts, keywords)

            # 计算聚类规模
            cluster_size = len(cluster_texts)

            # 选择有代表性的原始文本（非预处理版本）
            representative_texts = self._get_meaningful_representative_texts(original_texts)

            topic_analysis[cluster_id] = {
                'size': cluster_size,
                'keywords': keywords,
                'topic_label': topic_label,
                'representative_texts': representative_texts,
                'sample_texts': original_texts[:3]  # 原始文本样本
            }

        # 按聚类规模排序
        topic_analysis = dict(sorted(topic_analysis.items(),
                                     key=lambda x: x[1]['size'], reverse=True))

        return topic_analysis

    def _get_meaningful_representative_texts(self, texts: List[str], top_n: int = 3) -> List[str]:
        """
        获取有意义的代表性文本
        Args:
            texts: 文本列表
            top_n: 返回的代表性文本数量
        Returns:
            代表性文本列表
        """
        if not texts:
            return []

        # 过滤掉过短或无意义的文本
        meaningful_texts = []
        for text in texts:
            clean_text = str(text).strip()
            if (len(clean_text) >= 4 and  # 至少4个字符
                    not clean_text.isspace() and
                    not all(c in '的了呢吗啊呀哦' for c in clean_text)):
                meaningful_texts.append(clean_text)

        if not meaningful_texts:
            return []

        # 按长度和内容质量排序（优先选择中等长度、有实质内容的文本）
        meaningful_texts.sort(key=lambda x: (
            -len(x),  # 长度优先
            len([c for c in x if '\u4e00' <= c <= '\u9fff'])  # 中文字符数量
        ))

        return meaningful_texts[:top_n]

    def _get_representative_texts(self, cluster_id: int, cluster_texts: List[str]) -> List[str]:
        """
        获取代表性文本（聚类中心附近的文本）
        Args:
            cluster_id: 聚类ID
            cluster_texts: 聚类中的文本列表
        Returns:
            代表性文本列表
        """
        try:
            cluster_embeddings = self.embeddings[self.cluster_labels == cluster_id]
            cluster_center = cluster_embeddings.mean(axis=0)

            # 计算每个文本到聚类中心的距离
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)

            # 选择距离最近的3个文本作为代表性文本
            closest_indices = distances.argsort()[:3]
            representative_texts = [cluster_texts[i] for i in closest_indices]

            return representative_texts
        except:
            return cluster_texts[:3]  # 如果出错，返回前3个文本

    def visualize_clusters(self, save_path: str = None):
        """
        可视化聚类结果
        Args:
            save_path: 图片保存路径
        """
        if self.embeddings is None or self.cluster_labels is None:
            raise ValueError("请先生成嵌入向量并进行聚类")

        # 使用PCA降维到2D进行可视化
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)

        plt.figure(figsize=(12, 8))

        # 创建颜色映射
        unique_clusters = set(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:  # 噪声点
                color = 'gray'
                label = 'Noise'
                alpha = 0.3
            else:
                color = colors[i]
                label = f'Topic {cluster_id}'
                alpha = 0.7

            mask = self.cluster_labels == cluster_id
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[color], label=label, alpha=alpha, s=30)

        plt.title('Chat Topics Clustering Visualization', fontsize=16, fontweight='bold')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化图已保存至: {save_path}")

        plt.show()

    def generate_report(self, topic_analysis: Dict, save_path: str = None):
        """
        生成主题分析报告 - 改进版本
        Args:
            topic_analysis: 主题分析结果
            save_path: 报告保存路径
        """
        report = []
        report.append("=" * 80)
        report.append("                      聊天主题分析报告")
        report.append("=" * 80)
        report.append(f"总聊天记录数: {len(self.df):,}")
        report.append(f"分析出的主题数量: {len(topic_analysis)}")
        report.append("")

        for topic_id, analysis in topic_analysis.items():
            report.append(f"【主题 {topic_id}】{analysis['topic_label']}")
            report.append(f"  - 规模: {analysis['size']:,} 条消息 ({analysis['size'] / len(self.df) * 100:.1f}%)")
            report.append(f"  - 关键词: {', '.join(analysis['keywords'][:6])}")
            report.append(f"  - 代表性内容:")
            for i, text in enumerate(analysis['representative_texts'][:3], 1):
                if len(text) > 60:
                    text = text[:57] + "..."
                report.append(f"      {i}. {text}")
            report.append("")

        report_text = '\n'.join(report)
        print(report_text)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"分析报告已保存至: {save_path}")

        return report_text

    def save_topic_details(self, save_path: str = None):
        """
        保存每句话对应的主题明细到CSV文件
        Args:
            save_path: CSV文件保存路径
        """
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("请先进行聚类分析")

        # 创建主题明细DataFrame
        topic_details = self.df[['msg', 'processed_msg', 'cluster']].copy()

        # 获取主题分析结果来添加主题标签
        topic_analysis = self.analyze_topics()
        topic_label_map = {}
        for topic_id, analysis in topic_analysis.items():
            topic_label_map[topic_id] = analysis['topic_label']

        # 添加主题标签说明
        def get_topic_label(cluster_id):
            if cluster_id == -2:
                return "无效文本(预处理后为空)"
            elif cluster_id == -1:
                return "噪声点"
            else:
                return topic_label_map.get(cluster_id, f"主题{cluster_id}")

        topic_details['topic_label'] = topic_details['cluster'].apply(get_topic_label)

        # 重新排列列顺序
        topic_details = topic_details[['msg', 'processed_msg', 'cluster', 'topic_label']]

        if save_path is None:
            save_path = 'topic_details.csv'

        topic_details.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"主题明细已保存至: {save_path}")

        # 统计各主题分布
        topic_distribution = topic_details['topic_label'].value_counts()
        print("\n主题分布统计:")
        for topic, count in topic_distribution.items():
            percentage = count / len(topic_details) * 100
            print(f"  {topic}: {count:,} 条 ({percentage:.1f}%)")

        return topic_details


"""
主函数：执行完整的主题分析流程
"""
if __name__ == "__main__":
    # 初始化分析器
    analyzer = ChatTopicAnalyzer(model_name='BAAI/bge-base-zh')

    # 1. 加载数据
    csv_file = '2025-10-16_presto_80479171.csv'
    df = analyzer.load_data(csv_file)

    # 2. 生成嵌入向量
    print("正在生成文本嵌入向量...")
    analyzer.generate_embeddings()

    # 3. 聚类分析
    print("正在进行文本聚类...")
    analyzer.cluster_texts(method='kmeans', n_clusters=20)  # 可以调整聚类数量

    # 4. 主题分析（使用混合方法提取关键词）
    print("正在分析主题...")
    topic_analysis = analyzer.analyze_topics(keyword_method='hybrid')

    # 5. 保存主题明细CSV文件
    print("正在保存主题明细...")
    topic_details = analyzer.save_topic_details('topic_details.csv')

    # 6. 可视化
    print("生成可视化图表...")
    analyzer.visualize_clusters('topic_clusters.png')

    # 7. 生成报告
    print("生成分析报告...")
    analyzer.generate_report(topic_analysis, 'topic_analysis_report.txt')

    print("\n主题分析完成！")