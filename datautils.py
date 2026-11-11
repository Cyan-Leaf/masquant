import pdb
import os

_WIKITEXT_LOCAL_DIR = os.environ.get(
    "MASQUANT_WIKITEXT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext")
)


def _load_wikitext2_split(split):
    """优先从本地 parquet 加载 wikitext2，避免离线环境下访问 Hub 失败。"""
    local = os.path.join(_WIKITEXT_LOCAL_DIR, f"{split}-00000-of-00001.parquet")
    if os.path.exists(local):
        return load_dataset("parquet", data_files={split: local}, split=split)
    return load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
import transformers


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def compute_sqnr(original, quantized):
    error = original - quantized
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean(error ** 2)
    sqnr = 10 * torch.log10(signal_power / noise_power)
    
    return sqnr.item()

def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    traindata = _load_wikitext2_split('train')
    testdata = _load_wikitext2_split('test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_wikitext2_new(nsamples, seed, seqlen, model):
    print("get_wikitext2 new")
    traindata = _load_wikitext2_split('train')
    testdata = _load_wikitext2_split('test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        mask = trainenc.attention_mask[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        # trainloader.append((inp, tar))
        trainloader.append({"input_ids": inp, "attention_mask": mask, "labels": tar})
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    """评测用的 C4 只需要 validation；本地有 json.gz 就离线读，避免联网。
    用途：作为与 wikitext 不同领域的 PPL 测试集，检验混合校准的收益是否只是同域匹配（见报告 §2.1.4）。"""
    print("get_c4")
    import glob as _glob
    _local_val = os.environ.get("MASQUANT_C4_VAL") or ""
    if not _local_val:
        _cands = sorted(_glob.glob(os.path.join(os.environ.get("MASQUANT_DATA_ROOT", "./data"),
                                                "c4", "c4-validation*.json.gz")))
        _local_val = _cands[0] if _cands else ""
    if _local_val and os.path.exists(_local_val):
        print(f"load local c4 validation: {_local_val}")
        valdata = load_dataset('json', data_files=_local_val, split='train')
        traindata = None  # 只评测，不需要 train
    else:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    if traindata is not None:
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata  = load_dataset('ptb_text_only', 'penn_treebank', split='test')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    print("get_c4_new")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='',
):
    if 'wikitext2' in name:
        if transformers.__version__ == "4.31.0":
            print("当前 transformers 版本是 4.31.0, 使用旧的构造函数")
            return get_wikitext2(nsamples, seed, seqlen, model)   
        else:
            print(f"当前 transformers 版本是 {transformers.__version__}， 使用新的wikitext函数")     
            return get_wikitext2_new(nsamples, seed, seqlen, model)        
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)  
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)  
        return get_c4(nsamples, seed, seqlen, model)
    if 'mix' in name:
        wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model)
        ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model)
        c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model)
        train=wiki_train+ptb_train+c4_train
        val=None
        return train,val
