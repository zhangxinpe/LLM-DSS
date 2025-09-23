import os
import torch
from numpy.distutils.misc_util import all_strings
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Caption Generation
def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template


class TwitterDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tweets = df['tweet_content'].values
        self.targets = df['target'].values
        self.labels = df['sentiment'].values
        self.image_ids = df['image_id'].values
        self.ocr_text = df['ocr_text'].values
        self.od_result = df['od_result'].values
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.tweets)

    def all_strings(self, arr):
        return all(isinstance(item, str) for item in arr)

    def tokenizer_one(self, text):
        return self.tokenizer.encode_plus(
            text,  # 句子
            max_length=12,
            add_special_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
        )

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        ocr_text = str(self.ocr_text[item])
        od_result = eval(self.od_result[item])
        # print(self.od_result)
        # if self.od_result[item]=="['no text']":
        #     print("yes")
        #
        #
        #     if od_result[0]=="no text":
        #         print("yes")
        #         od_result=[]

        od_text_anps = ""
        #
        od_text_caption = []
        # if ocr_text == "no text":
        #     ocr_text = ""
        # else:
        #     od_text_caption.append(ocr_text)

        top_od_len = 3
        for i in range(len(od_result)):
            if i == top_od_len:  # range 从 0开始 为 0，1，2
                break
            od_text_anps = od_text_anps + " " + od_result[i][0]
            od_text_caption.append(od_result[i][3])
        # if not self.all_strings(od_text_caption):
        #     print(od_text_caption)

        if len(od_text_caption) <= 3:
            od_text_caption.append("no text")
            od_text_caption.append("no text")
            od_text_caption.append("no text")
            od_text_caption = od_text_caption[:3]
        od_text_caption = od_text_caption[:3]

        tweet_all = tweet + " " + ocr_text + " " + od_text_anps
        # tweet = tweet + ocr_text
        # tweet = tweet + od_result
        # tweet = tweet  + ocr_caption
        # tweet=tweet[:60]

        label = self.labels[item]
        target = self.targets[item]
        image_id = self.image_ids[item]
        image = Image.open(os.path.join(self.cfg.data_dir, self.cfg.dataset + "_images", image_id))
        image_clip = Image.open(os.path.join(self.cfg.data_dir, self.cfg.dataset + "_images", image_id))
        # image process
        image = self.transform(image)
        image = self.create_nested_tensor(image.unsqueeze(0))
        # get default caption and capmask
        start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        caption, cap_mask = create_caption_and_mask(start_token, 128)
        caption = caption.reshape(-1, )
        cap_mask = cap_mask.reshape(-1, )
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

        encoding_all = self.tokenizer.encode_plus(
            tweet_all,
            text_pair=target,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        # text-only encoding
        tweet_remove = tweet.replace('$T$', '')
        if tweet_remove == '':
            tweet_remove = target
        encoding_text = self.tokenizer.encode_plus(
            tweet_remove,
            max_length=100,
            add_special_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
        )

        encoding_ocr = self.tokenizer.encode_plus(
            ocr_text,
            max_length=32,
            add_special_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
        )

        # aspect-only encoding
        encoding_aspect = self.tokenizer.encode_plus(
            target,
            max_length=12,
            add_special_tokens=False,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
        )

        sentences = od_text_caption
        if len(sentences) != 3:
            print(sentences)
            print("yes")

        try:

            input_ids_1=self.tokenizer_one(sentences[0])
            attention_mask_1=self.tokenizer_one(sentences[0])
            input_ids_2=self.tokenizer_one(sentences[1])
            attention_mask_2=self.tokenizer_one(sentences[1])
            input_ids_3=self.tokenizer_one(sentences[2])
            attention_mask_3=self.tokenizer_one(sentences[2])
        except Exception as e:
            print("yes")
            print(e)
            print(sentences)

        # remove the last sep
        # encoding = remove_last_sep(encoding)
        # try:
        #     encoding_od_text = self.tokenizer.batch_encode_plus(
        #         od_text_caption,
        #         max_length = 12,
        #         add_special_tokens=False,
        #         padding="max_length",
        #         return_tensors="pt",
        #         return_token_type_ids=False,
        #         return_attention_mask=True,
        #         truncation=True,

        #     )
        # except Exception as e:
        #     print(e)
        #     print(od_text_caption)

        # print(od_text_caption)
        # print(encoding_od_text["input_ids"])

        return {
            "review_text": tweet,
            "targets": target,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "input_ids_all": encoding_all['input_ids'].flatten(),
            "attention_mask_all": encoding_all['attention_mask'].flatten(),
            "image": image.tensors.squeeze(0),
            "image_mask": image.mask.squeeze(0),
            "caption": caption,
            "caption_mask": cap_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "input_ids_tt": encoding_text["input_ids"].flatten(),
            "attention_mask_tt": encoding_text["attention_mask"].flatten(),
            "input_ids_at": encoding_aspect["input_ids"].flatten(),
            "attention_mask_at": encoding_aspect["attention_mask"].flatten(),
            "input_ids_ocr": encoding_ocr['input_ids'].flatten(),
            'attention_mask_ocr': encoding_ocr['attention_mask'].flatten(),
            "input_ids_1":input_ids_1['input_ids'].flatten(),
            "attention_mask_1":attention_mask_1['attention_mask'].flatten(),
            "input_ids_2":input_ids_2['input_ids'].flatten(),
            "attention_mask_2":attention_mask_2['attention_mask'].flatten(),
            "input_ids_3":input_ids_3['input_ids'].flatten(),
            "attention_mask_3":attention_mask_3['attention_mask'].flatten(),
            # "input_ids_od_text":input_ids,
            # "attention_mask_od_text":attention_mask,
        }


if __name__ == "__main__":

    import warnings
    import argparse

    warnings.filterwarnings("ignore")

    from torch.utils.data import DataLoader, Dataset

    from catr.datasets import coco, utils
    from catr.configuration import Config
    from transformers import AutoTokenizer, RobertaModel, BertModel

    from data_utils import *

    # Config
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--data_dir', type=str, default="/home/zxp/learn/datasets/IJCAI2019_data")
    parser.add_argument('--catr_ckpt', type=str, default="/home/zxp/learn/models/cart/checkpoint.pth")
    parser.add_argument('--result_dir', type=str, default="result")
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--dataset', type=str, default="twitter2015_caption")  # twitter2015 or twitter2017
    # parser.add_argument('--model', type=str, default="bertweet-base") # "bert-base-uncased" "vinai/bertweet-base"
    parser.add_argument('--model', type=str,
                        default="/home/zxp/learn/models/bertweet-base")  # "bert-base-uncased" "vinai/bertweet-base"
    parser.add_argument('--num_cycles', type=float, default=0.5)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--adamw_correct_bias', type=bool, default=True)
    parser.add_argument('--scheduler', type=str, default='linear')  # ['linear', 'cosine']
    parser.add_argument('--print_freq', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--max_grad_norm', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_catr', type=float, default=2e-5)
    parser.add_argument('--lr_backbone', type=float, default=5e-6)
    parser.add_argument('--max_caption_len', type=int, default=12)
    parser.add_argument('--max_len', type=int, default=128)  # default=72
    parser.add_argument('--sample_k', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--clip_image_feature_dim', type=int, default=512)
    parser.add_argument('--clip_text_feature_dim', type=int, default=512)

    args = parser.parse_args()

    device = torch.device(args.device)

    train_df, val_df, test_df = load_data(args)

    # Tokenizer
    # print(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.tokenizer = tokenizer
    args.pad_token_id = args.tokenizer.pad_token_id
    args.end_token_id = args.tokenizer.sep_token_id

    # Create DataLoader
    image_captions = None
    train_dataset = TwitterDataset(args, train_df, image_captions, coco.val_transform,
                                   utils.nested_tensor_from_tensor_list)
    test_dataset = TwitterDataset(args, test_df, image_captions, coco.val_transform,
                                  utils.nested_tensor_from_tensor_list)
    val_dataset = TwitterDataset(args, val_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    for step, d in enumerate(train_loader):
        input_ids_all = d["input_ids_all"].to(device)
        attention_mask_all = d["attention_mask_all"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        image = d["image"]
        image_mask = d["image_mask"]
        caption = d["caption"].to(device)
        cap_mask = d["caption_mask"].to(device)
        labels = d["labels"].to(device)
        input_ids_tt = d["input_ids_tt"].to(device)
        attention_mask_tt = d["attention_mask_tt"].to(device)
        input_ids_at = d["input_ids_at"].to(device)
        attention_mask_at = d["attention_mask_at"].to(device)

        # input_ids_1=d["input_ids_1"].to(device)
        # attention_mask_1=d["attention_mask_1"].to(device)
        # input_ids_2=d["input_ids_2"].to(device)
        # attention_mask_2=d["attention_mask_2"].to(device)
        # input_ids_3=d["input_ids_3"].to(device)
        # attention_mask_3=d["attention_mask_3"].to(device)
        # input_ids_od_text = torch.cat([input_ids_1,input_ids_2,input_ids_3],dim=0).to(device)
        # attention_mask_od_text = torch.cat([attention_mask_1,attention_mask_2,attention_mask_3],dim=0).to(device)

    pass



