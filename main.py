import sys, os, time, math, random
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from utils.data_loader import prepare_data_seq
from utils import config
from model.common_layer import count_parameters, make_infinite, evaluate
from model.trainer import Train_MIME


torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1234)

data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

if(config.test):
    print("Test model",config.model)
    model = Train_MIME(vocab, decoder_number=program_number, model_file_path=config.saved_model_path, is_eval=True)
    if (config.USE_CUDA):
        model.cuda()
    model = model.eval()

    loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b, bleu_score_t, ref_results = evaluate(model, data_loader_tst, ty="test",
                                                                               max_dec_step=50, write_summary=True)

    file_summary = config.save_path + "output.txt"
    with open(file_summary, 'w') as the_file:
        the_file.write("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\tBleu_t\n")
        the_file.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format("test", loss_test, ppl_test, acc_test, bleu_score_g, bleu_score_b, bleu_score_t))
        for o in ref_results: the_file.write(o)
    exit(0)


model = Train_MIME(vocab, decoder_number=program_number)
for n, p in model.named_parameters():
    if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
        xavier_uniform_(p)
print("TRAINABLE PARAMETERS", count_parameters(model))

check_iter = 1000
try:
    if (config.USE_CUDA):
        model.cuda()
    model = model.train()
    best_ppl = 1000
    patient = 0
    writer = SummaryWriter(log_dir=config.save_path)
    weights_best = deepcopy(model.state_dict())
    data_iter = make_infinite(data_loader_tra)
    for n_iter in tqdm(range(1000000)):
        temp = next(data_iter)
        loss, ppl, bce, acc = model.train_one_batch(temp, n_iter)
        writer.add_scalars('loss', {'loss_train': loss}, n_iter)
        writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
        writer.add_scalars('bce', {'bce_train': bce}, n_iter)
        writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
        if (config.noam):
            writer.add_scalars('lr', {'learning_rata': model.optimizer._rate}, n_iter)

        if ((n_iter + 1) % check_iter == 0):
            model = model.eval()
            model.epoch = n_iter
            model.__id__logger = 0
            loss_val, ppl_val, bce_val, acc_val, bleu_score_g, bleu_score_b, bleu_score_t = evaluate(model, data_loader_val,
                                                                                       ty="valid", max_dec_step=50)
            writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
            writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
            writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
            model = model.train()
            if (config.model == "mimic" and n_iter < 10000):
                continue
            if (ppl_val <= best_ppl):
                best_ppl = ppl_val
                patient = 0
                model.save_model(best_ppl, n_iter, 0, 0, bleu_score_g, bleu_score_b, bleu_score_t)
                weights_best = deepcopy(model.state_dict())
            else:
                patient += 1
            if (patient > 2): break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

## TESTING
model.load_state_dict({name: weights_best[name] for name in weights_best})
model.eval()
model.epoch = 1
loss_test, ppl_test, bce_test, acc_test, bleu_score_g, bleu_score_b, bleu_score_t, ref_results = evaluate(model, data_loader_tst, ty="test",
                                                                               max_dec_step=50, write_summary=True)

file_summary = config.save_path + "output.txt"
with open(file_summary, 'w') as the_file:
    the_file.write("### COMMENT: If using vader, score > 0 will be treated as positive emotion and score < 0 will be treated as negative emotion ### \n")
    the_file.write("EVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\tBleu_t\n")
    the_file.write(
        "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format("test", loss_test, ppl_test, acc_test, bleu_score_g,
                                                              bleu_score_b, bleu_score_t))
    
    for o in ref_results: the_file.write(o)
