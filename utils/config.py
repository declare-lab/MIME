import os
import logging 
import argparse

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6

bert_model = 'bert-base-uncased'
gpt2_model = 'gpt2'

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="empathetic")

parser.add_argument("--emo_input", type=str, default="self_att") # cross_att; self_att
parser.add_argument("--emo_combine", type=str, default="gate") # att; gate
parser.add_argument("--decoder", type=str, default="single") # single
parser.add_argument("--saved_model_path", type=str, default=None) # this arg is deprecated, use save_path instead
parser.add_argument("--vae", type=bool, default=False) # whether to use vae randomness and to add in the vae loss
parser.add_argument("--eq6_loss", type=bool, default=False) 
parser.add_argument("--vader_loss", type=bool, default=False) # add vader loss
parser.add_argument("--init_emo_emb", action="store_true")
 
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="save/test/")
parser.add_argument("--save_path_dataset", type=str, default="save/")
parser.add_argument("--cuda", default=True, action="store_true")

parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--basic_learner", default=True, action="store_true")
parser.add_argument("--project", action="store_true")
parser.add_argument("--topk", type=int, default=0)
parser.add_argument("--l1", type=float, default=.0)
parser.add_argument("--softmax", default=True, action="store_true")
parser.add_argument("--mean_query", action="store_true")
parser.add_argument("--schedule", type=float, default=10000)


parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", default=True, action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="mimic")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", default=True, action="store_true")
parser.add_argument("--noam", default=True, action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)

parser.add_argument("--emb_file", type=str)

## transformer 
parser.add_argument("--hop", type=int, default=1)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

arg = parser.parse_args()
print_opts(arg)
model = arg.model
dataset = arg.dataset
large_decoder = arg.large_decoder
topk = arg.topk
l1 = arg.l1
oracle = arg.oracle
basic_learner = arg.basic_learner
multitask = arg.multitask
softmax = arg.softmax
mean_query = arg.mean_query
schedule = arg.schedule
# Hyperparameters
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr
beam_size=arg.beam_size
project=arg.project
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm
# >>>>>>>>>> OUR ARGS 
emo_input = arg.emo_input
emo_combine = arg.emo_combine
decoder = arg.decoder
saved_model_path = arg.saved_model_path
vae = arg.vae
eq6_loss = arg.eq6_loss
vader_loss = arg.vader_loss
init_emo_emb = arg.init_emo_emb
# <<<<<<<<<<
USE_CUDA = arg.cuda
pointer_gen = arg.pointer_gen
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = 10000

emb_file = arg.emb_file or "vectors/glove.6B.{}d.txt".format(str(emb_dim))
pretrain_emb = arg.pretrain_emb

save_path = arg.save_path
save_path_dataset = arg.save_path_dataset

test = arg.test

### transformer 
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter


label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight

if test:
    emo_input = 'self_att'
    emo_combine = 'gate'
    # emo_combine = 'gate'
    model = 'mimic'
    label_smoothing = True
    noam = True
    emb_dim = 300
    emb_file = arg.emb_file or "vectors/glove.6B.{}d.txt".format(str(emb_dim))
    hidden_dim = 300
    hop = 1
    head = 2
    topk = 5
    pretrain_emb = False
    softmax = True
    basic_learner = True
    schedule = 10000
    saved_model_path = 'save/saved_model'


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))
collect_stats = False
