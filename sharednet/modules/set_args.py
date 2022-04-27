import argparse


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args() -> argparse.Namespace:
    """Get arguments/hyper-parameters for the experiment.

    Returns:
        Args instance

    """
    parser = argparse.ArgumentParser(prog="Sharednet",
                                     description='Shared network for different datasets',
                                     epilog="If you need any help, contact jiajingnan2222@gmail.com")

    parser.add_argument('--model_names', type=str,
                        help="'lobe_all', 'lobe_all_single', 'lobe_lu', 'lobe_ll', 'lobe_ru', 'lobe_rm', 'lobe_rl', "
                             "'vessel', 'AV_artery', 'AV_vein', ‘AV_all', 'liver', 'pancreas' ",
                        default='liver-pancreas')
    parser.add_argument('--mode', help='mode', type=str, choices=('train', 'infer'), default='train')
    parser.add_argument('--infer_data_dir', help='data directory for inference', type=str, default='')
    # parser.add_argument('--infer_weights_fpath', help='trained weights full path for inference', type=str, default='')
    parser.add_argument('--infer_ID', help='experiment ID of trained weights for inference', type=int, default=0)

    parser.add_argument('--loss', help='loss function', type=str, default='dice')

    parser.add_argument('--cond_flag', help='if conditioning or not', type=boolean_string, default=False)
    parser.add_argument('--cond_method', help='conditioining method', type=str, choices=('concat', 'mul_add'),
                        default='concat')
    parser.add_argument('--cond_pos', help='condition position', type=str, choices=('input', 'enc', 'dec', 'enc_dec'),
                        default='input')
    parser.add_argument('--same_mask_value', help='mask values for different tasks', type=boolean_string, default=True)

    parser.add_argument('--base', help='channel number of the first conv layer', type=int, default=32)
    parser.add_argument('--steps', help='training epochs', type=int, default=100001)
    parser.add_argument('--valid_period', help='period for validation', type=int, default=2000)

    parser.add_argument('--lr', help='learning rate for lobe segmentation', type=float, default=0.0001)
    parser.add_argument('--weight_decay', help='weight_decay', type=float, default=0.0001)

    parser.add_argument('--psz', help='patch_size', type=str, default="128_128")

    parser.add_argument('--cache', help='if cache dataset', type=boolean_string, default=True)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=1)
    parser.add_argument('--pps', help='patches per scan', type=int, default=10)
    parser.add_argument('--amp', help='mixed precision', type=boolean_string, default=True)
    parser.add_argument('--grad_accu', help='grad_accu', type=boolean_string, default=True)

    parser.add_argument('--outfile', help='output file when running by script instead of pycharm', type=str)
    parser.add_argument('--hostname', help='hostname of the server', type=str)
    parser.add_argument('--remark', help='comments on this experiment', type=str, default='amp')
    parser.add_argument('--jobid', help='slurm job_id', type=int, default=0)

    args = parser.parse_args()
    if args.model_names == 'lobe_all_single':
        args.model_names = 'lobe_lu-lobe_ll-lobe_ru-lobe_rm-lobe_rl'

    return args
