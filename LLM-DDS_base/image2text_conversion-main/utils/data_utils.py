import os
import pandas as pd
import emoji

def ocr_text_len(ocr_text,min_len,max_len):
    ocr_len=len(ocr_text.split(" "))
    if ocr_len >= min_len and ocr_len <= max_len:
        return ocr_text
    else:
        return "no text"



def load_data(args, replace_emoji=True):
    # data loading
    train_tsv = os.path.join(args.data_dir, args.dataset, "train.tsv")
    dev_tsv = os.path.join(args.data_dir, args.dataset, "dev.tsv")
    test_tsv = os.path.join(args.data_dir, args.dataset, "test.tsv")

    test_df = pd.read_csv(test_tsv, sep="\t")
    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(dev_tsv, sep="\t")


    if replace_emoji:
        train_df['tweet_content'] = train_df['tweet_content'].apply(emoji.replace_emoji)
        val_df['tweet_content'] = val_df['tweet_content'].apply(emoji.replace_emoji)
        test_df['tweet_content'] = test_df['tweet_content'].apply(emoji.replace_emoji)

        train_df["image_info"] = train_df["image_info"].apply(emoji.replace_emoji)
        val_df["image_info"] = val_df["image_info"].apply(emoji.replace_emoji)
        test_df["image_info"] = test_df["image_info"].apply(emoji.replace_emoji)

        # train_df["Aspect_Term_Background_Information"] = train_df["Aspect_Term_Background_Information"].apply(emoji.replace_emoji)
        # val_df["Aspect_Term_Background_Information"] = val_df["Aspect_Term_Background_Information"].apply(emoji.replace_emoji)
        # test_df["Aspect_Term_Background_Information"] = test_df["Aspect_Term_Background_Information"].apply(emoji.replace_emoji)

        # train_df["Aspect_Tweet_Relationship_and_Purpose"] = train_df["Aspect_Tweet_Relationship_and_Purpose"].apply(emoji.replace_emoji)
        # val_df["Aspect_Tweet_Relationship_and_Purpose"] = val_df["Aspect_Tweet_Relationship_and_Purpose"].apply(emoji.replace_emoji)
        # test_df["Aspect_Tweet_Relationship_and_Purpose"] = test_df["Aspect_Tweet_Relationship_and_Purpose"].apply(emoji.replace_emoji)

        

    # train_df["ocr_text"] = train_df["ocr_text"].apply(lambda x: ocr_text_len(x, args.ocr_min_len, args.ocr_max_len))
    # val_df["ocr_text"] = val_df["ocr_text"].apply(lambda x: ocr_text_len(x, args.ocr_min_len, args.ocr_max_len))
    # test_df["ocr_text"] = test_df["ocr_text"].apply(lambda x: ocr_text_len(x, args.ocr_min_len, args.ocr_max_len))

    return train_df, val_df, test_df



if __name__ == "__main__":
    load_data()
    pass
# import os
# import pandas as pd
# from emoji import demojize
# from nltk.tokenize import TweetTokenizer
# import json
#
#
# tokenizer = TweetTokenizer()
#
#
# def normalizeToken(token):
#     lowercased_token = token.lower()
#     if token.startswith("@"):
#         return "@USER"
#     elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
#         return "HTTPURL"
#     elif len(token) == 1:
#         return demojize(token)
#     else:
#         if token == "’":
#             return "'"
#         elif token == "…":
#             return "..."
#         else:
#             return token
#
#
# def normalizeTweet(tweet):
#     tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
#     normTweet = " ".join([normalizeToken(token) for token in tokens])
#
#     normTweet = (
#         normTweet.replace("cannot ", "can not ")
#         .replace("n't ", " n't ")
#         .replace("n 't ", " n't ")
#         .replace("ca n't", "can't")
#         .replace("ai n't", "ain't")
#     )
#     normTweet = (
#         normTweet.replace("'m ", " 'm ")
#         .replace("'re ", " 're ")
#         .replace("'s ", " 's ")
#         .replace("'ll ", " 'll ")
#         .replace("'d ", " 'd ")
#         .replace("'ve ", " 've ")
#     )
#     normTweet = (
#         normTweet.replace(" p . m .", "  p.m.")
#         .replace(" p . m ", " p.m ")
#         .replace(" a . m .", " a.m.")
#         .replace(" a . m ", " a.m ")
#     )
#
#     return " ".join(normTweet.split())
#
# def load_data(args, replace_emoji=True):
#     # data loading
#     train_tsv = os.path.join(args.data_dir, args.dataset, "train.tsv")
#     dev_tsv = os.path.join(args.data_dir, args.dataset, "dev.tsv")
#     test_tsv = os.path.join(args.data_dir, args.dataset, "test.tsv")
#
#     test_df = pd.read_csv(test_tsv, sep="\t")
#     train_df = pd.read_csv(train_tsv, sep="\t")
#     val_df = pd.read_csv(dev_tsv, sep="\t")
#
#
#     if replace_emoji:
#         train_df['tweet_content'] = train_df['tweet_content'].apply(normalizeTweet)
#         val_df['tweet_content'] = val_df['tweet_content'].apply(normalizeTweet)
#         test_df['tweet_content'] = test_df['tweet_content'].apply(normalizeTweet)
#
#
#     return train_df, val_df, test_df
#
# if __name__ == "__main__":
#     load_data()
#     pass
