import argparse

def parse_args_kgvae():
    parser = argparse.ArgumentParser(description="KGCL")
    # ===== log ===== #
    parser.add_argument('--desc', type=str, default="", help='EXP description')
    parser.add_argument('--log', action='store_true', default=False, help='log in file or not')
    parser.add_argument('--log_fn', type=str, default=None, help='log file_name')
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="movielens-100k", help="Choose a dataset:[last-fm, movielens-20m]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument('--model', default="KGCL", help='use MAE or not')
    # ===== train vae===== #
    parser.add_argument('--vae_epochs', type=int, default=50, help='number of vae pretrained epochs')
    parser.add_argument("--n_enc_epochs", type=int, default=3, help="Number of encoder-only update steps per RecVAE pretraining epoch")
    parser.add_argument("--n_dec_epochs", type=int, default=1, help="Number of decoder-only update steps per RecVAE pretraining epoch")

    # ===== train kgcl===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument('--sim_metric', choices=['cosine', 'wasserstein'], default='cosine', help='Stability metric: "cosine" or "wasserstein"')
    parser.add_argument("--node_dropout", type=int, default=1, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.2, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=int, default=1, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=int, default=1, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10]', help="List of K values for Top-K evaluation")
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    # parser.add_argument('--alpha'args_config.alpha)
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    parser.add_argument('--mu', type=float, default=0.7, help='mu')
    parser.add_argument('--tau', type=float, default=0.2, help='nu')
    parser.add_argument('--cl_weight', type=float, default=0.1, help='alpha')
    parser.add_argument(
        '--kg_ratio', type=float, default=1.0,
        help='Fraction of KG triples to use, range [0.1, 1.0]'
    )
    # ===== save model ===== #
    parser.add_argument("--save", action='store_true', default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()

if __name__ == "__main__":
    #
    args = parse_args_kgvae()

    #
    print(f"Experiment Description: {args.desc}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of Epochs: {args.epoch}")
    print(f"Using GPU: {args.cuda}, GPU ID: {args.gpu_id}")
    print(f"Save Model: {args.save}, Output Directory: {args.out_dir}")