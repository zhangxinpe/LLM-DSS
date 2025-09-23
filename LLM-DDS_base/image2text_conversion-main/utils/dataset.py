import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# 禁用警告
# 禁用警告
import warnings

warnings.filterwarnings("ignore")

import torch
import json  # 确保导入 json 库

def extract_words_by_distance(syntactic_distances, k):
    extracted_words = []
    word_instance_counts = {} # 用于跟踪单词实例计数

    for item in syntactic_distances: # 修正: 遍历字典列表
        token_text = item['token_text']
        distance = item['distance']
        if isinstance(distance, (int, float)) and distance <= k: # 修正: 确保 distance 是数字类型再比较
            word = token_text
            instance_index = word_instance_counts.get(word, -1) + 1 # 获取当前实例索引 (从 0 开始)
            word_instance_counts[word] = instance_index # 更新单词实例计数
            extracted_words.append({
                'word': word,
                'instance_index': instance_index, # 实例索引
                'distance': distance
            })
    return extracted_words


def extract_words_from_subtree(dependency_subtree):
    extracted_words = []
    word_instance_counts = {} # 用于跟踪单词实例计数

    for subtree_item in dependency_subtree:
        word = subtree_item['token_text'] # 修正: 从字典中获取 token_text
        relation = subtree_item['relation_to_parent']

        instance_index = word_instance_counts.get(word, -1) + 1 # 获取当前实例索引 (从 0 开始)
        word_instance_counts[word] = instance_index # 更新单词实例计数

        extracted_words.append({
            'word': word,
            'instance_index': instance_index, # 实例索引
            'relation_to_parent': relation
        })
    return extracted_words

def create_word_attention_mask(tokenizer,tokens_sentence_encoded_plus, words_instances):
    tokens_sentence_encoded = tokens_sentence_encoded_plus['input_ids'].tolist()[0]

    word_token_indices = {}

    for word_instance_dict in words_instances: # 修改: 遍历字典列表
        word = word_instance_dict['word'] # 修改: 从字典中获取 'word'
        instance_index = word_instance_dict['instance_index'] # 修改: 从字典中获取 'instance_index'
        word_instance = (word, instance_index) # 为了复用之前的逻辑，构建元组键

        tokens_word_encoded = tokenizer.encode(word, add_special_tokens=False)
        word_token_start_indices = []
        instance_count = 0 # 记录当前词语已找到的实例数量

        for i in range(len(tokens_sentence_encoded) - len(tokens_word_encoded) + 1):
            if tokens_sentence_encoded[i:i+len(tokens_word_encoded)] == tokens_word_encoded:
                if instance_count == instance_index: # 只关注指定实例索引的词语
                    word_token_start_indices.append(i)
                    break # 找到指定实例后就可以停止查找该词语的后续实例
                instance_count += 1 # 增加实例计数

        if word_token_start_indices:
            word_token_indices[word_instance] = [] # 键使用 (word, instance_index) 元组
            for start_index in word_token_start_indices:
                indices = list(range(start_index, start_index + len(tokens_word_encoded)))
                word_token_indices[word_instance].extend(indices)
        else:
            # print(f"警告: 单词 '{word}' (实例 {instance_index}) 没有在句子 token 序列中找到匹配.")
            pass

    attention_mask = [0] * len(tokens_sentence_encoded) # 初始化 attention_mask 为 0

    for word_instance, indices in word_token_indices.items(): # 遍历修改后的 word_token_indices
        for index in indices:
            attention_mask[index] = 1 # 将指定单词实例对应的 token 位置设为 1

    attention_mask[0] = 1 # 仍然保留 [CLS] 不掩码

    return torch.tensor(attention_mask)




class TwitterDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tweets = df['tweet_content'].values
        self.targets = df['target'].values
        self.labels = df['sentiment'].values
        self.image_ids = df['image_id'].values
        self.ocr_text = df['ocr_text'].values
        self.image_info = df['image_info'].values
        self.Aspect_Sentiment_Analysis = df['Aspect_Sentiment_Analysis'].values
        self.Aspect_Tweet_Relationship_and_Purpose = df['Aspect_Tweet_Relationship_and_Purpose'].values
        self.Aspect_Term_Background_Information = df["Aspect_Term_Background_Information"].values
        self.syntactic_distances = df['syntactic_distances'].values
        self.dependency_subtrees = df['dependency_subtree'].values
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        ocr_text = str(self.ocr_text[item])
        image_id = str(self.image_ids[item])
        target = str(self.targets[item])
        image_info = self.image_info[item]
        syntactic_distance = eval(self.syntactic_distances[item])
        dependency_subtree = eval(self.dependency_subtrees[item])

        # if isinstance(syntactic_distance, str):  # 检查是否为字符串，避免重复解析
        #     syntactic_distance = json.loads(syntactic_distance)  # 将 JSON 字符串转换为 Python 对象 (列表)
        # if isinstance(dependency_subtree, str):  # 检查是否为字符串，避免重复解析
        #     dependency_subtree = json.loads(dependency_subtree)  # 将 JSON 字符串转换为 Python 对象 (列表)

        Aspect_Term_Background_Information = str(self.Aspect_Term_Background_Information[item])
        Aspect_Sentiment_Analysis = str(self.Aspect_Sentiment_Analysis[item])
        Aspect_Tweet_Relationship_and_Purpose = str(self.Aspect_Tweet_Relationship_and_Purpose[item])

        if ocr_text != "no text":
            tweet_all = tweet + " " + ocr_text
        else:
            tweet_all = tweet

        tweet_all = tweet_all + ". " + image_info

        # tweet_all = tweet_all[:125]

        background =  Aspect_Term_Background_Information+ " ." + Aspect_Tweet_Relationship_and_Purpose
        background = background[:128]

        label = self.labels[item]
        target = self.targets[item]

        # text-aspect encoding (with special tokens)
        encoding = self.tokenizer.encode_plus(
            tweet,
            text_pair=target,
            add_special_tokens=True,
            max_length=60,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

                # 提取句法距离小于等于 k 的词
        extracted_by_distance_1 = extract_words_by_distance(syntactic_distance,1)
        extracted_by_distance_2 = extract_words_by_distance(syntactic_distance,2)
        extracted_by_distance_3 = extract_words_by_distance(syntactic_distance,3)
        extracted_by_distance_4 = extract_words_by_distance(syntactic_distance,4)
        extracted_by_distance_5 = extract_words_by_distance(syntactic_distance,5)

            # 提取依存子树中的词
        extracted_from_subtree = extract_words_from_subtree(dependency_subtree)


        # 使用 extract_words_by_distance 的输出创建 attention mask

        attention_mask_distance_1 = create_word_attention_mask(self.tokenizer,encoding, extracted_by_distance_1)
        attention_mask_distance_2 = create_word_attention_mask(self.tokenizer,encoding, extracted_by_distance_2)
        attention_mask_distance_3 = create_word_attention_mask(self.tokenizer,encoding, extracted_by_distance_3)
        attention_mask_distance_4 = create_word_attention_mask(self.tokenizer,encoding, extracted_by_distance_4)
        attention_mask_distance_5 = create_word_attention_mask(self.tokenizer,encoding, extracted_by_distance_5)

        # print(attention_mask_distance)



        # 使用 extract_words_from_subtree 的输出创建 attention mask
        attention_mask_subtree = create_word_attention_mask(self.tokenizer,encoding, extracted_from_subtree)
        # print(attention_mask_subtree)




        # text-aspect encoding (with special tokens)





        encoding_all = self.tokenizer.encode_plus(
            tweet_all,
            text_pair=target,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation="longest_first",
        )

        encoding_background = self.tokenizer.encode_plus(
            background,
            text_pair=target,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation="longest_first",
        )

        return {
            "review_text": tweet,
            "targets": target,
            "image_id": image_id,
            "ocr_text": ocr_text,
            "image_info": image_info,
            "Aspect_Term_Background_Information": Aspect_Term_Background_Information,
            "Aspect_Sentiment_Analysis": Aspect_Sentiment_Analysis,
            "Aspect_Tweet_Relationship_and_Purpose": Aspect_Tweet_Relationship_and_Purpose,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "input_ids_all": encoding_all['input_ids'].flatten(),
            "attention_mask_all": encoding_all['attention_mask'].flatten(),
            "input_ids_background": encoding_background['input_ids'].flatten(),
            "attention_mask_background": encoding_background['attention_mask'].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
            "attention_mask_distance_1": attention_mask_distance_1.flatten(),
            "attention_mask_distance_2": attention_mask_distance_2.flatten(),
            "attention_mask_distance_3": attention_mask_distance_3.flatten(),
            "attention_mask_distance_4": attention_mask_distance_4.flatten(),
            "attention_mask_distance_5": attention_mask_distance_5.flatten(),
            "attention_mask_subtree": attention_mask_subtree.flatten()
            
        }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pass



