import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RS .")

    # shared parameter
    parser.add_argument('--epoch', type=int, default=700,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Output sizes of every layer')
    parser.add_argument('--dataset', nargs='?', default='amazonbook',
                        help='Choose a dataset from {gowalla, yelp2018, amazonbook}')
    parser.add_argument('--model', nargs='?', default='LightGCN',
                        help='model: [BPRMF, NCF, NGCF, LightGCN, LGNGuard, AMF]')
    parser.add_argument('--data_path', nargs='?', default='./dataset/',
                        help='Input data path.')
    parser.add_argument('--neg_ratio', type=int, default=1,
                        help='negtive sampling ratio')
    parser.add_argument('--loss_type', type=str, default='bpr', 
                        help='[bpr, bce, apr]')

    # NCF parameter
    parser.add_argument("--dropout_rate", type=float,default=0.0,  
                        help="dropout rate")
    parser.add_argument("--NCF_layers", type=int,default=3, 
	                    help="number of layers in MLP module in NCF model")
    parser.add_argument('--NCF_model_name', nargs='?', default='NeuMF',
                        help='model: [GMF, MLP, NeuMF]')      
    
    # APR parameter
    parser.add_argument('--reg_adv', type=float, default=0.1,
                        help='Regularization for adversarial loss')
    parser.add_argument('--adv_epoch', type=int, default=500,
                        help='Add APR in epoch X, when adv_epoch is 0, it\'s equivalent to pure AMF.\n '
                             'And when adv_epoch is larger than epochs, it\'s equivalent to pure MF model. ')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon for adversarial weights.')
    
    # 
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('-f')
    return parser.parse_args()
args = parse_args()
