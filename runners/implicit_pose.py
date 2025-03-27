import os
import logging
import time
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np

from models.gcnpose import GCNpose, adj_mx_from_edges
from models.igcn import IGCN
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps
from common.data_utils import fetch_me, read_3d_data_me, create_2d_data
from common.generators import PoseGenerator_gmm
from common.loss import mpjpe, p_mpjpe
from common.memory_utils import get_dynamic_chunk_size, adaptive_chunk_processing

class Implicitpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).to(device)
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # Adaptive configuration
        self.use_implicit = getattr(self.args, "use_implicit", False)
        
        # Dynamic chunk sizing configuration
        self.use_dynamic_chunks = getattr(self.args, "use_dynamic_chunks", False)
        self.process_chunk_size = getattr(self.args, "process_chunk_size", 256)
        self.min_chunk_size = getattr(self.args, "min_chunk_size", 32)
        self.max_chunk_size = getattr(self.args, "max_chunk_size", 1024)
        
        # Set memory configs if provided in command line
        if hasattr(config, 'implicit') and not hasattr(config.implicit, 'min_chunk_size'):
            config.implicit.min_chunk_size = self.min_chunk_size
            config.implicit.max_chunk_size = self.max_chunk_size
            config.implicit.target_memory_usage = getattr(self.args, "target_memory_usage", 0.75)
        
        # Performance metrics tracking
        self.track_metrics = getattr(self.args, "track_metrics", False)
        if self.track_metrics:
            self.inference_times = []
            self.iteration_counts = []
            self.memory_usage = []
            self.chunk_sizes_used = []

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            
            # Ensure same data processing
            stride = self.args.downsample
            logging.info(f'Using downsample stride: {stride}')
            
            self.dataset = read_3d_data_me(dataset)
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
                print('==> Selected actions: {}'.format(self.action_filter))
                
            # Log dataset size information for comparison
            poses_train, poses_train_2d, actions_train, camerapara_train = \
                fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
            
            # Debug for dataset preprocessing
            print(f"[IPOSE DEBUG] === Dataset sizes with stride={stride} ===")
            print(f"[IPOSE DEBUG] Number of sequences: {len(poses_train)}")
            print(f"[IPOSE DEBUG] Total frames: {sum([p.shape[0] for p in poses_train])}")
            print(f"[IPOSE DEBUG] poses_train: Length={len(poses_train)}, First 3 shapes={[p.shape for p in poses_train[:3]]}")
            print(f"[IPOSE DEBUG] poses_train_2d: Length={len(poses_train_2d)}, First 3 shapes={[p.shape for p in poses_train_2d[:3]]}")
            print(f"[IPOSE DEBUG] actions_train: Length={len(actions_train)}, First 3 lengths={[len(a) for a in actions_train[:3]]}")
            print(f"[IPOSE DEBUG] camerapara_train: Length={len(camerapara_train)}")
            print(f"[IPOSE DEBUG] Batch size: {config.training.batch_size}")
            print(f"[IPOSE DEBUG] Expected iterations per epoch: {sum([p.shape[0] for p in poses_train]) // config.training.batch_size}")
                
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
                
            # Report total frames to process
            train_frames = sum([poses_train[i].shape[0] for i in range(len(poses_train))])
            test_frames = sum([poses_valid[i].shape[0] for i in range(len(poses_valid))])
            logging.info(f'Training dataset: {len(poses_train)} sequences, {train_frames} total frames')
            logging.info(f'Testing dataset: {len(poses_valid)} sequences, {test_frames} total frames')
            
            # Calculate batches to process
            batch_size = config.training.batch_size
            train_batches = train_frames // batch_size + (1 if train_frames % batch_size != 0 else 0)
            test_batches = test_frames // batch_size + (1 if test_frames % batch_size != 0 else 0)
            logging.info(f'With batch size {batch_size}: {train_batches} training batches, {test_batches} testing batches')
        else:
            raise KeyError('Invalid dataset')

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        
        # Use adaptive model if specified
        if self.use_implicit:
            logging.info('Using Implicit GCN (warm start OFF by default)')
            self.model_diff = IGCN(adj.to(self.device), config).to(self.device)
        else:
            from models.gcndiff import GCNdiff
            logging.info('Using standard GCNdiff')
            self.model_diff = GCNdiff(adj.to(self.device), config).to(self.device)
            
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
            
        # After model creation, update chunk size dynamically if enabled
        if self.use_dynamic_chunks and hasattr(self, 'model_diff') and torch.cuda.is_available():
            # Create a dummy batch for testing (small size)
            dummy_batch_size = min(16, self.min_chunk_size)
            dummy_batch = torch.zeros((dummy_batch_size, 17, 5), device=self.device)
            dummy_t = torch.ones(dummy_batch_size, device=self.device)
            
            # Determine optimal chunk size
            self.process_chunk_size = get_dynamic_chunk_size(
                self.model_diff.module, self.config, self.device, 
                test_input=dummy_batch, mask=self.src_mask
            )
            logging.info(f"Initial dynamic chunk size: {self.process_chunk_size}")
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_pose = GCNpose(adj.cuda(), config).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states[0])
        else:
            logging.info('initialize model randomly')

    def process_batch_adaptive(self, x, mask, t, model_fn):
        """
        Process a batch with adaptive chunk sizes based on current memory conditions
        
        Args:
            x: Input tensor
            mask: Attention mask
            t: Timestep tensor
            model_fn: Function to call on each chunk
            
        Returns:
            Processed output
        """
        # If dynamic chunking is disabled, use fixed chunk size
        if not self.use_dynamic_chunks:
            return self.process_batch_in_chunks(x, mask, t, self.process_chunk_size, model_fn)
        
        # Define processing function for adaptive_chunk_processing
        def process_chunk(chunk, chunk_mask, chunk_t):
            return model_fn(chunk, chunk_mask, chunk_t)
        
        # Use the adaptive processing utility
        result = adaptive_chunk_processing(
            self.model_diff.module, 
            x, mask, t, 
            process_fn=process_chunk,
            max_chunk_size=self.max_chunk_size,
            device=self.device
        )
        
        return result
        
    def process_batch_in_chunks(self, x, mask, t, chunk_size, model_fn):
        """
        Process a batch in fixed-size chunks
        
        Args:
            x: Input tensor
            mask: Attention mask
            t: Timestep tensor
            chunk_size: Size of chunks to process
            model_fn: Function to call on each chunk
            
        Returns:
            Processed output
        """
        if x.size(0) <= chunk_size:
            return model_fn(x, mask, t)
            
        batch_size = x.size(0)
        outputs = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Get current chunk - use clone to avoid in-place issues
            chunk = x[i:end_idx].clone()
            
            # Handle mask
            if mask.size(0) == 1:
                chunk_mask = mask.expand(end_idx - i, -1, -1)
            else:
                chunk_mask = mask[i:end_idx].clone()
            
            # Handle timestep
            if isinstance(t, torch.Tensor) and t.size(0) > 1:
                chunk_t = t[i:end_idx].clone()
            else:
                chunk_t = t
            
            # Process chunk
            outputs.append(model_fn(chunk, chunk_mask, chunk_t))
            
            # Free memory
            if i % (chunk_size * 2) == 0 and i > 0:
                torch.cuda.empty_cache()
                
        # Combine outputs
        return torch.cat(outputs, dim=0)

    def train(self):
        # Enable anomaly detection during debugging
        # torch.autograd.set_detect_anomaly(True)
        
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_p1, best_epoch = 1000, 0
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset')
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
        
        # Tracking metrics for adaptive model
        train_iterations = []
        train_times = []
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            # Reset history between epochs if using implicit model
            if self.use_implicit and hasattr(self.model_diff.module, 'reset_history'):
                self.model_diff.module.reset_history()
                    
            # Clear GPU cache between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Re-determine chunk size at each epoch if using dynamic chunks
            if self.use_dynamic_chunks and torch.cuda.is_available():
                self.process_chunk_size = get_dynamic_chunk_size(
                    self.model_diff.module, self.config, self.device)
                logging.info(f"Epoch {epoch} dynamic chunk size: {self.process_chunk_size}")
            
            epoch_loss_diff = AverageMeter()

            for i, (targets_uvxyz, targets_noise_scale, _, targets_3d, _, _) in enumerate(data_loader):
                data_time += time.time() - data_start
                step += 1

                # Define model function for diffusion training
                def process_model_batch(chunk_x, chunk_mask, chunk_t):
                    return self.model_diff(chunk_x, chunk_mask, chunk_t.float(), 0)
                
                # Process with smaller chunks to prevent OOM issues
                train_start = time.time()
                
                # Move data to device - use clone to avoid in-place issues
                targets_uvxyz = targets_uvxyz.to(self.device)
                targets_noise_scale = targets_noise_scale.to(self.device)
                targets_3d = targets_3d.to(self.device)
                
                # Generate noisy sample
                n = targets_3d.size(0)
                x = targets_uvxyz.clone()
                e = torch.randn_like(x)
                b = self.betas
                
                # Create random timesteps
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                # Apply noise
                e = e * targets_noise_scale.clone()
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                
                # Generate x_t (refer to DDIM equation) - avoid in-place operations
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                # Process with adaptive or fixed-size chunking
                if self.use_dynamic_chunks:
                    output_noise = self.process_batch_adaptive(x, src_mask, t, process_model_batch)
                else:
                    output_noise = self.process_batch_in_chunks(
                        x, src_mask, t, self.process_chunk_size, process_model_batch)
                
                # Calculate loss with new tensor creation
                loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)
                
                # Update parameters
                optimizer.zero_grad()
                loss_diff.backward()  # This is where the in-place error was occurring
                torch.nn.utils.clip_grad_norm_(self.model_diff.parameters(), config.optim.grad_clip)
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), targets_3d.size(0))
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                # End training timing
                train_end = time.time()
                train_times.append(train_end - train_start)
                
                # Track iteration count if using implicit model
                if self.use_implicit and hasattr(self.model_diff.module, 'last_iteration_count'):
                    train_iterations.append(self.model_diff.module.last_iteration_count)
                
                if i%100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
                    
                    # Log computational metrics for implicit model
                    if self.use_implicit and train_iterations and len(train_iterations) > 0:
                        avg_iterations = sum(train_iterations) / len(train_iterations)
                        avg_time = sum(train_times) / len(train_times)
                        logging.info(f'| Avg Iterations: {avg_iterations:.2f} | Avg Train Time: {avg_time:.4f}s | Chunks: {len(train_iterations)} |')
                        train_iterations = []
                        train_times = []
                        
                # Optimize memory if needed
                if self.use_implicit and i % 1000 == 0 and hasattr(self.model_diff.module, 'optimize_memory'):
                    self.model_diff.module.optimize_memory()
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                
            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            
                logging.info('test the performance of current model')

                p1, p2 = self.test_hyber(is_train=True)

                if p1 < best_p1:
                    best_p1 = p1
                    best_epoch = epoch
                logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(best_epoch, best_p1, epoch, p1, p2))
    
    def test_hyber(self, is_train=False):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False, 
                num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset') 

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        # Reset history for implicit model
        if self.use_implicit and hasattr(self.model_diff.module, 'reset_history'):
            self.model_diff.module.reset_history()
        
        # Reset performance tracking metrics
        if self.track_metrics:
            self.inference_times = []
            self.iteration_counts = []
            self.memory_usage = []
            self.chunk_sizes_used = []
            
        # Re-determine chunk size for testing
        if self.use_dynamic_chunks and torch.cuda.is_available():
            self.process_chunk_size = get_dynamic_chunk_size(
                self.model_diff.module, self.config, self.device)
            logging.info(f"Testing with dynamic chunk size: {self.process_chunk_size}")
        
        # Clear GPU cache before testing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',\
            'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        action_error_sum = define_error_list(self.test_action_list)        

        for i, (_, input_noise_scale, input_2d, targets_3d, input_action, camera_para) in enumerate(data_loader):
            data_time += time.time() - data_start

            input_noise_scale, input_2d, targets_3d = \
                input_noise_scale.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

            # build uvxyz
            inputs_xyz = self.model_pose(input_2d, src_mask)            
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :] 
            input_uvxyz = torch.cat([input_2d, inputs_xyz], dim=2)
                        
            # Generate repeated inputs for test_times
            full_size = input_uvxyz.size(0) * test_times
            input_uvxyz_repeated = input_uvxyz.repeat(test_times, 1, 1)
            input_noise_scale_repeated = input_noise_scale.repeat(test_times, 1, 1)
            
            # Define model function for inference
            def process_inference_batch(chunk_uvxyz, chunk_mask, chunk_t):
                if self.use_implicit:
                    # Direct inference with implicit model
                    return self.model_diff(chunk_uvxyz, chunk_mask, chunk_t)
                else:
                    # Standard diffusion approach
                    x = chunk_uvxyz.clone()
                    # Skip noise addition steps for efficiency
                    return generalized_steps(x, chunk_mask, seq, self.model_diff, self.betas, eta=self.args.eta)[0][-1]
            
            # Track memory before inference
            if self.track_metrics:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
            
            # Time the inference
            start_time = time.time()
            
            # Prepare timestep tensor
            t = torch.ones(input_uvxyz_repeated.size(0), device=self.device) * test_num_diffusion_timesteps
            
            # Process with adaptive chunking
            if self.use_dynamic_chunks:
                # Track what chunk sizes were used
                prev_chunk_size = self.process_chunk_size
                output_uvxyz = self.process_batch_adaptive(input_uvxyz_repeated, src_mask, t, process_inference_batch)
                if self.track_metrics:
                    self.chunk_sizes_used.append(self.process_chunk_size)
            else:
                # Fixed chunk size processing
                output_uvxyz = self.process_batch_in_chunks(
                    input_uvxyz_repeated, src_mask, t, self.process_chunk_size, process_inference_batch)
                
            # End timing
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Track metrics
            if self.track_metrics:
                self.inference_times.append(end_time - start_time)
                
                # Track memory usage
                torch.cuda.synchronize()
                mem_peak = torch.cuda.max_memory_allocated()
                self.memory_usage.append((mem_peak - mem_before) / (1024 * 1024))  # Convert to MB
                
                # Track iteration count for implicit model
                if self.use_implicit and hasattr(self.model_diff.module, 'last_iteration_count'):
                    self.iteration_counts.append(self.model_diff.module.last_iteration_count)
            
            # Reshape and average test_times samples
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times, -1, 17, 5), 0)
            
            # Process results
            output_xyz = output_uvxyz[:, :, 2:]
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]
            epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0, targets_3d.size(0))
            
            data_start = time.time()
            
            action_error_sum = test_calculation(output_xyz, targets_3d, input_action, action_error_sum, None, None)
            
            if i%100 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(data_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
                
                # Log computational metrics
                if self.track_metrics:
                    if len(self.inference_times) > 0:
                        avg_time = sum(self.inference_times) / len(self.inference_times)
                        logging.info(f'Average inference time: {avg_time:.4f}s')
                    
                    if self.use_implicit and hasattr(self.model_diff.module, 'last_iteration_count') and len(self.iteration_counts) > 0:
                        avg_iterations = sum(self.iteration_counts) / len(self.iteration_counts)
                        logging.info(f'Average iteration count: {avg_iterations:.2f}')
                    
                    if len(self.memory_usage) > 0:
                        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
                        logging.info(f'Average peak memory usage: {avg_memory:.2f} MB')
                        
                    if self.use_dynamic_chunks and len(self.chunk_sizes_used) > 0:
                        avg_chunk = sum(self.chunk_sizes_used) / len(self.chunk_sizes_used)
                        logging.info(f'Average chunk size: {avg_chunk:.0f}')
                
                # Optimize memory during test
                if self.use_implicit and hasattr(self.model_diff.module, 'optimize_memory'):
                    self.model_diff.module.optimize_memory()
                
        # Final computational metrics
        if self.track_metrics and len(self.inference_times) > 0:
            self.log_performance_metrics(os.path.join(self.args.log_path, "performance_metrics.txt"))
                
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(batch=i + 1, size=len(data_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                    e2=epoch_loss_3d_pos_procrustes.avg))
        
        p1, p2 = print_error(None, action_error_sum, is_train)

        return p1, p2
        
    def log_performance_metrics(self, output_path=None):
        """
        Log detailed performance metrics and optionally save to a file
        """
        if not self.track_metrics or not (self.inference_times and len(self.inference_times) > 0):
            return
            
        # Compute statistics
        avg_time = sum(self.inference_times) / len(self.inference_times)
        max_time = max(self.inference_times)
        min_time = min(self.inference_times)
        
        # Compute iterations statistics if available
        if self.use_implicit and hasattr(self.model_diff.module, 'last_iteration_count') and len(self.iteration_counts) > 0:
            avg_iters = sum(self.iteration_counts) / len(self.iteration_counts)
            max_iters = max(self.iteration_counts)
            min_iters = min(self.iteration_counts)
            iter_data = f"Iterations: avg={avg_iters:.2f}, min={min_iters}, max={max_iters}"
        else:
            iter_data = "Iterations: not tracked"
        
        # Compute memory statistics if available
        if len(self.memory_usage) > 0:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            max_memory = max(self.memory_usage)
            min_memory = min(self.memory_usage)
            mem_data = f"Memory (MB): avg={avg_memory:.2f}, min={min_memory:.2f}, max={max_memory:.2f}"
        else:
            mem_data = "Memory: not tracked"
            
        # Compute chunk size statistics if using dynamic chunks
        if self.use_dynamic_chunks and len(self.chunk_sizes_used) > 0:
            avg_chunk = sum(self.chunk_sizes_used) / len(self.chunk_sizes_used)
            max_chunk = max(self.chunk_sizes_used)
            min_chunk = min(self.chunk_sizes_used)
            chunk_data = f"Chunk sizes: avg={avg_chunk:.0f}, min={min_chunk}, max={max_chunk}"
        else:
            chunk_data = f"Fixed chunk size: {self.process_chunk_size}"
        
        # Log performance summary
        logging.info("=== Performance Summary ===")
        logging.info(f"Time (s): avg={avg_time:.4f}, min={min_time:.4f}, max={max_time:.4f}")
        logging.info(iter_data)
        logging.info(mem_data)
        logging.info(chunk_data)
        
        # Save metrics to file if path provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("=== Performance Metrics ===\n")
                f.write(f"Time (s): avg={avg_time:.4f}, min={min_time:.4f}, max={max_time:.4f}\n")
                f.write(f"{iter_data}\n")
                f.write(f"{mem_data}\n")
                f.write(f"{chunk_data}\n")
                f.write("\n=== Raw Data ===\n")
                f.write(f"Times: {self.inference_times}\n")
                if self.use_implicit and hasattr(self.model_diff.module, 'last_iteration_count'):
                    f.write(f"Iterations: {self.iteration_counts}\n")
                f.write(f"Memory: {self.memory_usage}\n")
                if self.use_dynamic_chunks:
                    f.write(f"Chunk sizes: {self.chunk_sizes_used}\n")