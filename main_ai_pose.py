import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.diffpose_frame import Diffpose


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=19960903, help="Random seed")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the config file")
    parser.add_argument("--exp", type=str, default="exp", 
                        help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, required=True, 
                        help="A string for documentation purpose. "\
                            "Will be the name of the log folder.", )
    parser.add_argument("--verbose", type=str, default="info", 
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    ### Diffformer configuration ####
    #Diffusion process hyperparameters
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quad(quadratic))")
    parser.add_argument("--eta", type=float, default=0.0, 
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")
    # Diffusion model parameters
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor')
    # load pretrained model
    parser.add_argument('--model_diff_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--model_pose_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--train', action = 'store_true',
                        help='train or evluate')
    #training hyperparameter
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--lr_gamma', default=0.9, type=float, metavar='N',
                        help='weight decay rate')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--decay', default=60, type=int, metavar='N',
                        help='decay frequency(epoch)')
    #test hyperparameter
    parser.add_argument('--test_times', default=5, type=int, metavar='N',
                    help='the number of test times')
    parser.add_argument('--test_timesteps', default=50, type=int, metavar='N',
                    help='the number of test time steps')
    parser.add_argument('--test_num_diffusion_timesteps', default=500, type=int, metavar='N',
                    help='the number of test times')
                    
    # Adaptive model options
    parser.add_argument('--use_adaptive', action='store_true',
                        help='Use adaptive implicit model instead of standard diffusion')
    parser.add_argument('--track_metrics', action='store_true',
                        help='Track computational metrics (time, memory, iterations)')
    parser.add_argument('--implicit_solver', type=str, default=None,
                       help='Solver for implicit model: anderson or fixed_point')
    parser.add_argument('--implicit_iters', type=int, default=None,
                       help='Maximum iterations for adaptive solver')
    parser.add_argument('--anderson_m', type=int, default=None,
                       help='History size for Anderson acceleration')
    parser.add_argument('--adaptive_alpha', action='store_true',
                       help='Use adaptive relaxation parameter')
    parser.add_argument('--warm_start', action='store_true',
                       help='Use warm starting between batches')
    parser.add_argument('--progressive_tol', action='store_true',
                       help='Use progressive tolerance schedule')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device
    # update configure file
    new_config.training.batch_size = args.batch_size
    new_config.optim.lr = args.lr
    new_config.optim.lr_gamma = args.lr_gamma
    new_config.optim.decay = args.decay
    
    # Set up implicit configuration if needed
    if args.use_adaptive:
        if not hasattr(new_config, 'implicit'):
            new_config.implicit = argparse.Namespace()
            # Default implicit configuration
            new_config.implicit.solver = "anderson"
            new_config.implicit.max_iterations = 20
            new_config.implicit.tolerance = 1e-5
            new_config.implicit.anderson_m = 5
            new_config.implicit.anderson_beta = 1.0
            new_config.implicit.anderson_lambda = 1e-4
            new_config.implicit.use_adaptive_alpha = True
            new_config.implicit.init_alpha = 0.5
            new_config.implicit.min_alpha = 0.1
            new_config.implicit.max_alpha = 0.9
            new_config.implicit.use_progressive_tol = True
            new_config.implicit.init_tol = 1e-3
            new_config.implicit.final_tol = 1e-5
            new_config.implicit.tol_decay_steps = 5000
            new_config.implicit.use_warm_start = True
            new_config.implicit.warm_start_momentum = 0.9
        
        # Override with command line arguments
        if args.implicit_solver is not None:
            new_config.implicit.solver = args.implicit_solver
        if args.implicit_iters is not None:
            new_config.implicit.max_iterations = args.implicit_iters
        if args.anderson_m is not None:
            new_config.implicit.anderson_m = args.anderson_m
        if args.adaptive_alpha:
            new_config.implicit.use_adaptive_alpha = True
        if args.warm_start:
            new_config.implicit.use_warm_start = True
        if args.progressive_tol:
            new_config.implicit.use_progressive_tol = True

    if args.train:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    # Log model type
    if args.use_adaptive:
        logging.info("Using adaptive implicit model")
        implicit_solver = getattr(config.implicit, 'solver', 'anderson')
        implicit_iters = getattr(config.implicit, 'max_iterations', 20)
        logging.info(f"Implicit solver: {implicit_solver}, Max iterations: {implicit_iters}")
        
        # Log adaptive settings
        logging.info("Adaptive settings:")
        logging.info(f"  Warm start: {getattr(config.implicit, 'use_warm_start', True)}")
        logging.info(f"  Adaptive relaxation: {getattr(config.implicit, 'use_adaptive_alpha', True)}")
        logging.info(f"  Progressive tolerance: {getattr(config.implicit, 'use_progressive_tol', True)}")
        
        # Log tolerance information
        if getattr(config.implicit, 'use_progressive_tol', True):
            init_tol = getattr(config.implicit, 'init_tol', 1e-3)
            final_tol = getattr(config.implicit, 'final_tol', 1e-5)
            steps = getattr(config.implicit, 'tol_decay_steps', 5000)
            logging.info(f"  Tolerance: {init_tol} → {final_tol} over {steps} steps")
        else:
            tol = getattr(config.implicit, 'tolerance', 1e-5)
            logging.info(f"  Tolerance: {tol}")
    else:
        logging.info("Using standard diffusion model")
    
    try:
        runner = Diffpose(args, config)
        runner.create_diffusion_model(args.model_diff_path)
        runner.create_pose_model(args.model_pose_path)
        runner.prepare_data()
        if args.train:
            runner.train()
        else:
            _, _ = runner.test_hyber()
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())