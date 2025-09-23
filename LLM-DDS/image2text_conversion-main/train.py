import gc
import warnings
from sklearn.metrics import f1_score
from transformers import AdamW
from transformers import AutoTokenizer, RobertaModel, BertModel
from model.MabsaModel import *
from model.OutModel import *
from model.text_model import *
from Loss.combin_loss import *
from utils.helpers import *
from utils.dataset import *
from utils.data_utils import *

warnings.filterwarnings("ignore")  # 禁用警告
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")
import re
import argparse


# Config
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda:3")
parser.add_argument('--data_dir', type=str, default="./datasets/2015数据集")
parser.add_argument('--result_dir', type=str, default="results")
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--dataset', type=str, default="twitter2015_gpt_all")  # twitter2015 or twitter2017
# parser.add_argument('--model', type=str, default="bertweet-base") # "bert-base-uncased" "vinai/bertweet-base"
parser.add_argument('--model', type=str,
                    default="./models/bertweet-base")  # "bert-base-uncased" "vinai/bertweet-base"
parser.add_argument('--adamw_correct_bias', type=bool, default=True)
parser.add_argument('--scheduler', type=str, default='linear')  # ['linear', 'cosine']
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--max_grad_norm', type=int, default=20)
parser.add_argument('--seed', type=int, default=21)

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--lr_fus_output', type=float, default=2e-5)
parser.add_argument('--max_len', type=int, default=196)  # default=72
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--bert_dim', type=int, default=768)
parser.add_argument('--save_dir', type=str, default="./results/xlsx2015")
args = parser.parse_args()

args = parser.parse_args()

device = torch.device(args.device)

best_acc = {"acc": 0, "macro_f1": 0}

from datetime import datetime  # 获取当前时间，格式化为 "年-月-日_小时-分钟-秒"

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 使用 time 和日志路径合并
# 使用 time 和日志路径合并
log_path = os.path.join(args.log_dir, f"base_loss3_local_only_ocr_twomodel_{current_time}_{args.dataset}")
LOGGER = get_logger(log_path)
train_df, val_df, test_df = load_data(args)

tokenizer = AutoTokenizer.from_pretrained(args.model)
args.tokenizer = tokenizer
args.pad_token_id = args.tokenizer.pad_token_id
args.end_token_id = args.tokenizer.sep_token_id
LOGGER.info(f"lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}, model={args.model}, dataset={args.dataset}")



## Train fn
def train_fn(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch, test_loader):
    model = model.train()
    losses = []
    correct_predictions = 0
    start = time.time()
    for step, d in enumerate(data_loader):
        input_ids_all = d["input_ids_all"].to(device)
        attention_mask_all = d["attention_mask_all"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        input_ids_background = d["input_ids_background"].to(device)
        attention_mask_background = d["attention_mask_background"].to(device)
        attention_mask_distance_1 = d["attention_mask_distance_1"].to(device)
        attention_mask_distance_2 = d["attention_mask_distance_2"].to(device)
        attention_mask_distance_3 = d["attention_mask_distance_3"].to(device)
        attention_mask_distance_4 = d["attention_mask_distance_4"].to(device)
        attention_mask_subtree = d["attention_mask_subtree"].to(device)

        attention_mask_fus = [attention_mask_distance_1, attention_mask_distance_2, attention_mask_distance_3, attention_mask_subtree,attention_mask]

        only_text_output, text_image_output, fus_output, weight1_value_numpy, weight2_value_numpy = model(
            input_ids, attention_mask,
            input_ids_all, attention_mask_all,
            input_ids_background,
            attention_mask_background,
            attention_mask_fus,
        )
        _, preds = torch.max(fus_output, dim=1)

        loss = loss_fn(only_text_output, text_image_output, fus_output, labels)

        correct_predictions += torch.sum(preds == labels).item()
        losses.append(loss.item())
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # clip grad
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % (args.print_freq ) == 0 or step == (len(data_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss:.4f} '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, step, len(data_loader),
                          remain=timeSince(start, float(step + 1) / len(data_loader)),
                          loss=loss.item(),
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0],
                          weight1_value_numpy=weight1_value_numpy.item(),
                          weight2_value_numpy=weight2_value_numpy.item()))


    return correct_predictions / n_examples, np.mean(losses)


# eval fn
import numpy as np
import pandas as pd


def format_eval_output(rows):
    # 解包数据，确保所有数据列正确分配
    tweets, targets, image_id, ocr_text, labels, predictions, image_info, Aspect_Term_Background_Information, Aspect_Tweet_Relationship_and_Purpose, Aspect_Sentiment_Analysis, only_text_output, text_image_output, fus_output = zip(
        *rows)

    # 通过 np.vstack 将每一列数据堆叠起来
    tweets = np.vstack(tweets) if isinstance(tweets[0], np.ndarray) else np.array(tweets)
    targets = np.vstack(targets) if isinstance(targets[0], np.ndarray) else np.array(targets)
    ocr_text = np.vstack(ocr_text) if isinstance(ocr_text[0], np.ndarray) else np.array(ocr_text)
    labels = np.vstack(labels) if isinstance(labels[0], np.ndarray) else np.array(labels)
    predictions = np.vstack(predictions) if isinstance(predictions[0], np.ndarray) else np.array(predictions)
    image_info = np.vstack(image_info) if isinstance(image_info[0], np.ndarray) else np.array(image_info)
    Aspect_Term_Background_Information = np.vstack(Aspect_Term_Background_Information) if isinstance(
        Aspect_Term_Background_Information[0], np.ndarray) else np.array(Aspect_Term_Background_Information)
    Aspect_Tweet_Relationship_and_Purpose = np.vstack(Aspect_Tweet_Relationship_and_Purpose) if isinstance(
        Aspect_Tweet_Relationship_and_Purpose[0], np.ndarray) else np.array(Aspect_Tweet_Relationship_and_Purpose)
    Aspect_Sentiment_Analysis = np.vstack(Aspect_Sentiment_Analysis) if isinstance(Aspect_Sentiment_Analysis[0],
                                                                                   np.ndarray) else np.array(
        Aspect_Sentiment_Analysis)

    # 将 logits 转换为 NumPy 数组
    only_text_output = np.vstack(only_text_output) if isinstance(only_text_output[0], np.ndarray) else np.array(
        only_text_output)
    text_image_output = np.vstack(text_image_output) if isinstance(text_image_output[0], np.ndarray) else np.array(
        text_image_output)
    fus_output = np.vstack(fus_output) if isinstance(fus_output[0], np.ndarray) else np.array(
        fus_output)

    # 将堆叠后的数据转换为 DataFrame
    results_df = pd.DataFrame({
        "tweet": tweets.reshape(-1).tolist(),
        "target": targets.reshape(-1).tolist(),
        "ocr_text": ocr_text.reshape(-1).tolist(),
        "image_id": image_id,
        "label": labels.reshape(-1).tolist(),
        "prediction": predictions.reshape(-1).tolist(),
        "image_info": image_info.reshape(-1).tolist(),
        "Aspect_Term_Background_Information": Aspect_Term_Background_Information.reshape(-1).tolist(),
        "Aspect_Tweet_Relationship_and_Purpose": Aspect_Tweet_Relationship_and_Purpose.reshape(-1).tolist(),
        "Aspect_Sentiment_Analysis": Aspect_Sentiment_Analysis.reshape(-1).tolist(),
        "only_text_output_0": only_text_output[:, 0].tolist(),  # 添加 only_text_output 的第一个logit
        "only_text_output_1": only_text_output[:, 1].tolist(),  # 添加 only_text_output 的第二个logit
        "only_text_output_2": only_text_output[:, 2].tolist(),  # 添加 only_text_output 的第三个logit
        "text_image_output_0": text_image_output[:, 0].tolist(),  # 添加 text_image_output 的第一个logit
        "text_image_output_1": text_image_output[:, 1].tolist(),  # 添加 text_image_output 的第二个logit
        "text_image_output_2": text_image_output[:, 2].tolist(),  # 添加 text_image_output 的第三个logit
        "fus_output_0": fus_output[:, 0].tolist(),  # 添加 fus_output 的第一个logit
        "fus_output_1": fus_output[:, 1].tolist(),  # 添加 fus_output 的第二个logit
        "fus_output_2": fus_output[:, 2].tolist(),  # 添加 fus_output 的第三个logit
    })

    return results_df


def eval_model(model, data_loader, loss_fn, device, n_examples, detailed_results=False):
    model = model.eval()
    losses = []
    correct_predictions = 0
    rows = []
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            input_ids_all = d["input_ids_all"].to(device)
            attention_mask_all = d["attention_mask_all"].to(device)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            input_ids_background = d["input_ids_background"].to(device)
            attention_mask_background = d["attention_mask_background"].to(device)
            attention_mask_distance_1 = d["attention_mask_distance_1"].to(device)
            attention_mask_distance_2 = d["attention_mask_distance_2"].to(device)
            attention_mask_distance_3 = d["attention_mask_distance_3"].to(device)
            attention_mask_distance_4 = d["attention_mask_distance_4"].to(device)
            attention_mask_subtree = d["attention_mask_subtree"].to(device)

            attention_mask_fus = [attention_mask_distance_1, attention_mask_distance_2, attention_mask_distance_3, attention_mask_subtree,attention_mask]

            only_text_output, text_image_output, fus_output, weight1_value_numpy, weight2_value_numpy = model(
                input_ids, attention_mask,
                input_ids_all, attention_mask_all,
                input_ids_background,
                attention_mask_background,
                attention_mask_fus,
            )
            _, preds = torch.max(fus_output, dim=1)

            loss = loss_fn(only_text_output, text_image_output, fus_output, labels)

            correct_predictions += torch.sum(preds == labels).item()
            losses.append(loss.item())
            rows.extend(
                zip(d["review_text"],
                    d["targets"],
                    d['image_id'],
                    d['ocr_text'],
                    d["labels"].numpy(),
                    preds.cpu().numpy(),
                    d["image_info"],
                    d["Aspect_Term_Background_Information"],
                    d["Aspect_Tweet_Relationship_and_Purpose"],
                    d['Aspect_Sentiment_Analysis'],
                    only_text_output.cpu().numpy(),  # 添加 only_text_output
                    text_image_output.cpu().numpy(),  # 添加 text_image_output
                    fus_output.cpu().numpy()  # 添加 fus_output
                    )
            )

        if detailed_results:
            return (correct_predictions / n_examples,
                    np.mean(losses),
                    format_eval_output(rows),
                    )

    return correct_predictions / n_examples, np.mean(losses)


def train_loop():
    LOGGER.info(f"========== Start Training ==========")
    LOGGER.info(f"lr={args.lr}, seed={args.seed},epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}, model={args.model}, dataset={args.dataset}")
    # Create DataLoader
    train_dataset = TwitterDataset(args, train_df)
    test_dataset = TwitterDataset(args, test_df)
    val_dataset = TwitterDataset(args, val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    image_text_model = TextImageModel(args)
    only_text_model = OnlyTextModel(args)
    model = FusedModel_loss_1_1(only_text_model, image_text_model)
    # print(model)
    model.to(device)
    # Configure the optimizer and scheduler.
    param_dicts = [
        {
            "params": model.stack_fus.parameters(),  # 只为 stack_fus 层设置不同的学习率
            "lr": args.lr_fus_output,
        },
        {
            "params": [param for name, param in model.named_parameters() if "stack_fus" not in name],  # 其余层使用默认学习率
            "lr": args.lr,
        }
    ]
    # param_dicts = model.parameters()

    optimizer = AdamW(param_dicts, lr=args.lr, correct_bias=args.adamw_correct_bias)

    num_train_steps = int(len(train_df) / args.batch_size * args.epochs)
    scheduler = get_scheduler(args, optimizer, num_train_steps)
    # 这个权重是每一类的样本数与总样本数的比例的倒数，这样定义loss的作用是对样本不均衡的类别给予更高的权重
    # loss_weights = torch.FloatTensor([3719/368,
    #                                     3179/1883,
    #                                     3179/928,])
    # loss_fn = nn.CrossEntropyLoss(weight=loss_weights).to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn = CombinedLoss_3_1().to(device)

    for epoch in range(args.epochs):
        print(f"===============Epoch {epoch + 1}/{args.epochs}==============")
        start_time = time.time()

        train_acc, train_loss = train_fn(
            model, train_loader, loss_fn, optimizer, device, scheduler, len(train_df), epoch, test_loader
        )

        val_acc, val_loss, dr = eval_model(model, val_loader, loss_fn, device, len(val_df), detailed_results=True)
        macro_f1 = f1_score(dr.label, dr.prediction, average="macro")
        LOGGER.info(f'Epoch {epoch + 1} - Val loss {val_loss} accuracy {val_acc} macro f1 {macro_f1}')

        test_acc, test_loss, dr = eval_model(model, test_loader, loss_fn, device, len(test_df), detailed_results=True)
        macro_f1 = f1_score(dr.label, dr.prediction, average="macro")
        dr.to_excel(f"results/epoch_{epoch + 1}_test.xlsx")

        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch + 1} - Train loss {train_loss} accuracy {train_acc}" time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Test loss {test_loss} accuracy {test_acc} macro f1 {macro_f1}')

    torch.cuda.empty_cache()
    gc.collect()
    LOGGER.info(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")
    return dr


if __name__ == "__main__":
    # best_acc = {"acc": 0, "macro_f1": 0}# args.seed = random.randint(0, 10000)
    # args.seed = 79
    # x = train_loop()
    # seed_everything(seed = args.seed)
    seed_everything(seed = args.seed)
    x = train_loop()



