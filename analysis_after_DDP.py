import os
import torch
import pickle
import argparse
from dynamic_bernoulli_embeddings.analysis import DynamicEmbeddingAnalysis
from dynamic_bernoulli_embeddings.embeddings import DynamicBernoulliEmbeddingModel

def analysis(path, dataset_dict, **kwargs):

    # setting gpu id of this process
    torch.cuda.set_device(0)

    # Build model.
    model = DynamicBernoulliEmbeddingModel(
        len(dataset_dict['data_class'].dictionary),
        dataset_dict['data_class'].T,
        dataset_dict['data_class'].m_t,
        dataset_dict['data_class'].dictionary,
        dataset_dict['data_class'].unigram_logits.cuda(),
        **kwargs,
    ).cuda()

    # pt file load
    pt = torch.load(path)
    ckpt, loss_history = pt['model'], pt['loss_history']

    # load ckpt to model
    model.load_state_dict(ckpt)

    # analysis
    embeddings = model.get_embeddings()
    emb = DynamicEmbeddingAnalysis(embeddings, dataset_dict['data_class'].dictionary)
    emb.absolute_drift()  # Terms that changed between the first and last timesteps
    emb.neighborhood("climate", 0)  # [input t!] Find nearby terms for "climate" at time `t`
    emb.change_points()

    # save
    with open('checkpoint/emb.pkl', 'wb') as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # fetch args
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--gpu', default='0', type=str, help='single GPU!')
    parser.add_argument('--ckpt-path', default="checkpoint/model.pt", type=str)
    parser.add_argument('--dataset-path', default='data/dataset_0.1.pkl', type=str)
    args = parser.parse_args()

    # cuda visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load
    with open(args.dataset_path, 'rb') as f: dataset_dict = pickle.load(f)

    analysis(args.ckpt_path, dataset_dict)
