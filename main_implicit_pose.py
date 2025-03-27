import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.implicit_pose import Implicitpose


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
    parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
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
                    
    # Implicit model options
    parser.add_argument('--use_implicit', action='store_true',
                        help='Use implicit model instead of standard diffusion')
    parser.add_argument('--track_metrics', action='store_true',
                        help='Track computational metrics (time, memory, iterations)')
    parser.add_argument('--implicit_iters', type=int, default=20,
                       help='Maximum iterations for implicit solver')
    parser.add_argument('--implicit_tol', type=float, default=1e-2,
                       help='Tolerance for implicit solver convergence')
    parser.add_argument('--min_iterations', type=int, default=10,
                       help='Minimum iterations for implicit solver')
    parser.add_argument('--use_warm_start', action='store_true',
                       help='Use warm starting between batches (OFF by default)')
    parser.add_argument('--use_memory_efficient', action='store_true',
                        help='Use memory-efficient attention to reduce GPU memory usage')
                        
    # Dynamic chunk sizing options
    parser.add_argument('--use_dynamic_chunks', action='store_true',
                        help='Dynamically determine optimal chunk size based on GPU memory')
    parser.add_argument('--process_chunk_size', type=int, default=256,
                        help='Fixed chunk size when dynamic chunks disabled')
    parser.add_argument('--min_chunk_size', type=int, default=32,
                        help='Minimum chunk size when using dynamic chunks')
    parser.add_argument('--max_chunk_size', type=int, default=1024,
                        help='Maximum chunk size when using dynamic chunks')
    parser.add_argument('--target_memory_usage', type=float, default=0.75,
                        help='Target GPU memory usage (0.0-1.0) for dynamic chunk sizing')
    
    # Memory optimization options
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='Enable PyTorch anomaly detection for debugging')
    parser.add_argument('--expandable_segments', action='store_true',
                        help='Enable PyTorch expandable segments for memory fragmentation')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device:', device)
    new_config.device = device
    # update configure file
    new_config.training.batch_size = args.batch_size
    new_config.optim.lr = args.lr
    new_config.optim.lr_gamma = args.lr_gamma
    new_config.optim.decay = args.decay
    
    # Set up implicit configuration if needed
    if args.use_implicit:
        if not hasattr(new_config, 'implicit'):
            new_config.implicit = argparse.Namespace()
            # Default implicit configuration
            new_config.implicit.max_iterations = args.implicit_iters
            new_config.implicit.tolerance = args.implicit_tol
            new_config.implicit.min_iterations = args.min_iterations
            new_config.implicit.solver = "anderson"
            new_config.implicit.anderson_m = 5
            new_config.implicit.anderson_beta = 1.0
            new_config.implicit.anderson_lambda = 1e-2
            new_config.implicit.use_warm_start = args.use_warm_start  # Default OFF unless specified
            new_config.implicit.warm_start_momentum = 0.5
            # Set up memory-efficient attention
            new_config.implicit.use_memory_efficient = args.use_memory_efficient
            
        # Add dynamic chunk configuration
        new_config.implicit.min_chunk_size = args.min_chunk_size
        new_config.implicit.max_chunk_size = args.max_chunk_size
        new_config.implicit.target_memory_usage = args.target_memory_usage

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

        # Clear existing handlers to prevent duplication
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
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
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
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
    # Configure PyTorch memory settings if requested
    args, config = parse_args_and_config()
    
    # Set up PyTorch anomaly detection if requested
    if hasattr(args, 'detect_anomaly') and args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("PyTorch anomaly detection enabled")
    
    # Set up expandable segments for memory fragmentation if requested
    if hasattr(args, 'expandable_segments') and args.expandable_segments:
        # This is handled via environment variable, but we'll log it
        print("PyTorch expandable segments should be enabled via environment variable")
        print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before running")
    
    # Pre-clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger = logging.getLogger()
    logger.info("Writing log file to {}".format(args.log_path))
    logger.info("Exp instance id = {}".format(os.getpid()))
    
    # Log configuration
    if args.use_memory_efficient:
        logger.info("Using memory-efficient attention to reduce GPU memory usage")
    
    if args.use_implicit:
        logger.info("Using implicit model")
        logger.info(f"Implicit parameters:")
        logger.info(f"  Max iterations: {args.implicit_iters}")
        logger.info(f"  Min iterations: {args.min_iterations}")
        logger.info(f"  Tolerance: {args.implicit_tol}")
        logger.info(f"  Warm start: {args.use_warm_start} (OFF by default)")
        logger.info(f"  Tracking metrics: {args.track_metrics}")
    else:
        logger.info("Using standard diffusion model")
        
    # Log dynamic chunking configuration
    if args.use_dynamic_chunks:
        logger.info("Using dynamic chunk sizing:")
        logger.info(f"  Min chunk size: {args.min_chunk_size}")
        logger.info(f"  Max chunk size: {args.max_chunk_size}")
        logger.info(f"  Target memory usage: {args.target_memory_usage*100:.0f}%")
    else:
        logger.info(f"Using fixed chunk size: {args.process_chunk_size}")
    
    # Ensure experiment directory exists
    os.makedirs(args.log_path, exist_ok=True)
    
    try:
        runner = Implicitpose(args, config)
        runner.create_diffusion_model(args.model_diff_path)
        runner.create_pose_model(args.model_pose_path)
        runner.prepare_data()
        if args.train:
            logger.info("Starting training...")
            runner.train()
        else:
            logger.info("Starting evaluation...")
            _, _ = runner.test_hyber()
    except Exception:
        logger.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())