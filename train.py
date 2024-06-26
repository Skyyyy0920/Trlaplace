import time
import random
import logging
import argparse
import collections
from turtle import pd
from tqdm import tqdm
from torchtext.vocab import Vectors
import torchtext
from datasets import load_dataset
from model import DAE, VAE, AAE
from vocab import Vocab
from meter import AverageMeter
from utils import *
from batchify import get_batches

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--dataset', type=str,
                    # default='sst2',
                    default='ag_news',
                    # default='imdb',
                    help='path to training file')
parser.add_argument('--save-dir', default='checkpoints', help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', help='path to load checkpoint if specified')

# Architecture arguments
parser.add_argument('--vocab-size', type=int, default=10000, help='keep N most frequent words in vocabulary')
parser.add_argument('--dim_z', type=int, default=128, help='dimension of latent variable z')
parser.add_argument('--dim_emb', type=int,
                    # default=768,
                    default=300,
                    help='dimension of word embedding')  # =20
parser.add_argument('--dim_h', type=int, default=512, help='dimension of hidden state per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--dim_d', type=int, default=512, help='dimension of hidden state in AAE discriminator')

# Model arguments
parser.add_argument('--model_type', default='dae', choices=['dae', 'vae', 'aae'],
                    help='which model to learn')
parser.add_argument('--lambda_kl', type=float, default=0, help='weight for kl term in VAE')
parser.add_argument('--lambda_adv', type=float, default=0, help='weight for adversarial loss in AAE')
parser.add_argument('--lambda_p', type=float, default=0, help='weight for L1 penalty on posterior log-variance')
parser.add_argument('--noise', default='0.1,0.1,0.1,0.1',
                    help='word drop prob, blank prob, substitute prob max word shuffle distance')
parser.add_argument('--method', default='trlap', choices=['gau', 'lap', 'trlap', 'mdp', 'maha', 'privemb'],
                    help='different private embedding methods LDP and Metric DP')
parser.add_argument('--eps', type=float,
                    # default=0.01,
                    # default=1,
                    # default=5,
                    default=10,
                    help='privacy budget epsilon')
parser.add_argument('--pretrained_vectors', type=str,
                    # default='wiki-news-300d-1M.vec',
                    # default='glove.6B.300d.txt',
                    default='random',
                    help='pretrained_vectors')

# Training arguments
parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
# parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=1024)
# Others
parser.add_argument('--seed', type=int, default=42, help='random seed')


def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets in batches:
            losses = model.autoenc(inputs, targets)
            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args.save_dir = f'checkpoints/{args.dataset}_{args.method}_{args.eps}_{args.pretrained_vectors}/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logging_config(args.save_dir)
    logging.info(str(args))

    train_sents = []
    if args.dataset == 'sst2':
        dataset = load_dataset(path='sst2', cache_dir='./data')
        for set in dataset:
            for row in dataset[set]:
                train_sents.append(row['sentence'].strip().split())
    elif args.dataset == 'ag_news':
        dataset = load_dataset(path='ag_news', cache_dir='./data')
        for set in dataset:
            for row in dataset[set]:
                train_sents.append(row['text'].strip().split())
        random.seed(42)
        train_sents = random.sample(train_sents, 16000)
        print(train_sents[:5])
    elif args.dataset == 'imdb':
        dataset = load_dataset(path='imdb', cache_dir='./data', ignore_verifications=True)
        for set in dataset:
            for row in dataset[set]:
                train_sents.append(row['text'].strip().split())
        random.seed(42)
        train_sents = random.sample(train_sents, 5000)
        print(train_sents[:5])
    else:
        raise ValueError("No such dataset")
    valid_sents = random.sample(train_sents, int(len(train_sents) * 0.1))
    logging.info('# train sents {}, tokens {}'.format(len(train_sents), sum(len(s) for s in train_sents)))
    logging.info('# valid sents {}, tokens {}'.format(len(valid_sents), sum(len(s) for s in valid_sents)))

    vocab_file = os.path.join('./vocabulary', f'{args.dataset}.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(train_sents, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)
    logging.info('# vocab size {}'.format(vocab.size))

    data = []
    f = open(vocab_file, 'r', encoding="utf-8")
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-1]
        data.append(line.split('\t'))
    df = pd.DataFrame(data)

    # Use pre-trained word embedding
    if args.pretrained_vectors != 'random':
        pretrained_vectors = Vectors(name=f'./embeddings/{args.pretrained_vectors}')
        weight_matrix = pretrained_vectors.get_vecs_by_tokens(list(df[0]))
    else:
        weight_matrix = torch.randn(len(df[0]), args.dim_emb)

    set_seed(args.seed)
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model_type](vocab, weight_matrix, args).to(device)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
    logging.info('# model parameters: {}'.format(sum(x.data.nelement() for x in model.parameters())))

    train_batches, _ = get_batches(train_sents, vocab, args.batch_size, device)
    valid_batches, _ = get_batches(valid_sents, vocab, args.batch_size, device)
    best_val_loss = None
    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        model.train()
        weight = model.embed.weight.cpu().detach().numpy()
        train_embed = np.column_stack((np.array(df[0]), weight))
        np.savetxt(os.path.join(args.save_dir, 'fine_tune.txt'), train_embed, delimiter=' ', fmt='%s', encoding='utf8')
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i_budget, idx in enumerate(indices):
            inputs, targets = train_batches[idx]
            losses = model.autoenc(inputs, targets, is_train=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i_budget + 1) % 100 == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(epoch + 1, i_budget + 1, len(indices))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging.info(log_output)

        valid_meters = evaluate(model, valid_batches)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            log_output += ' {} {:.2f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            # log_output += ' | saving model'
            ckpt = {'args': args, 'model': model.state_dict()}
            # torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            best_val_loss = valid_meters['loss'].avg
        logging.info(log_output)
    logging.info('Done training')


if __name__ == '__main__':
    args = parser.parse_args()
    args.noise = [float(x) for x in args.noise.split(',')]
    main(args)
