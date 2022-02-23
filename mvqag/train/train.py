from asyncio import subprocess
import torch
import pandas as pd
import torchmetrics
from collections import OrderedDict
from rich.progress import track
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List

from mvqag.utils import manage_log_files
from mvqag.data.data import DataLoaders
torch.autograd.set_detect_anomaly(True)


__all__ = [
    # Functions
    'get_device',

    # Classes
    'VQATrainer', 'TrainingLogger', 'Checkpointer',
]

# ============================= FUNCTIONS ==============================
def get_device():
    """Get torch device instance and available GPU IDs"""
    from torch.cuda import device_count
    from torch import device
    cuda_ids = [0] if device_count() == 1 else list(range(device_count()))
    return device(f"cuda:{cuda_ids[0]}"), cuda_ids


# ============================= CLASSES ==============================
class Checkpointer:
    def __init__(self,
                 chkpts_dir: Union[str, Path],
                 chkpt_of: Optional[List[Dict[str, str]]] = [
                     {'ValLoss': 'min'}],
                 chkpt_prefix: Optional[str] = None,
                 chkpt_after: int = 5) -> None:
        """Model's weights saving and loading handler

        Args:
            chkpts_dir: Folder path to save checkpoint files.
            chkpt_of: List of metrics to track. Each metric must be specified
                in a dict where key is the name of the metric (this metric
                must be present in trainer's metrics argument) and value is
                the improvement direction. For example, loss metrics improves
                on decreasing. Hence, {'loss': 'min'}.
            chkpt_prefix: First name of the checkpoint file. Last name will
                depend on the type of checkpoint.
            chkpt_after: Create a checkpoint after given number of epochs.
                If chkpt_after=1 then checkpoint will be created after every
                epoch.
        """
        self.chkpts_dir, self.chkpt_of = Path(chkpts_dir), chkpt_of
        self.chkpts_dir.mkdir(parents=True, exist_ok=True)
        self.chkpt_prefix = f"{chkpt_prefix}_" if chkpt_prefix else ''
        self.chkpt_after = chkpt_after
        # Save model weights on given metrics
        self.chkpts_handler = {}
        for _met_dict in self.chkpt_of:
            _met_name, _met_mode = list(_met_dict.items())[0]
            if _met_mode == 'min':
                self.chkpts_handler[_met_name] = float('inf')
            elif _met_mode == 'max':
                self.chkpts_handler[_met_name] = 0.
            else:
                _errmsg = "Either pass `min` to save weights when converge "
                _errmsg += "(for losses) or pass `max` to save weights when "
                _errmsg += "metric improves (such as accuracy)"
                raise ValueError(_errmsg)

    def create(self,
               epoch: int,
               net: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optm_fn: torch.nn.Module,
               savepath: Union[str, Path] = None) -> Path:
        """Create checkpoint of model and optimizer state"""
        if savepath is None:
            savepath = self.chkpts_dir/f'{self.chkpt_prefix}chkpt.ptc'
        torch.save({
            'net_state_dict': net.state_dict(),
            'optim_state_dict': optm_fn.state_dict(),
            'loss_fn': loss_fn,
            'global_epoch': epoch,
        }, savepath)
        print(f"Checkpoint created @ epoch {epoch}.")
        return savepath

    def load(self,
             path: Union[str, Path],
             net: torch.nn.Module,
             loss_fn: torch.nn.Module,
             optm_fn: torch.nn.Module,
             device: torch.device) -> Tuple[list, torch.nn.Module, int]:
        chkpt = torch.load(path, map_location=device)
        net_missing_keys = net.load_state_dict(chkpt['net_state_dict'])
        optm_missing_keys = optm_fn.load_state_dict(chkpt['optim_state_dict'])
        loss_fn = chkpt['loss_fn']
        global_epoch = chkpt['global_epoch']
        return [net_missing_keys, optm_missing_keys], loss_fn, global_epoch

    def save_best_wts(self, epoch_logs, net) -> None:
        """Save checkpoints when metric improves"""
        for _met_dict in self.chkpt_of:
            _met_name, _met_mode = list(_met_dict.items())[0]
            if _met_name not in epoch_logs.keys():
                _errmsg = f"Metric `{_met_name}` must match names in given"
                _errmsg += "metrics. Available metrics are "
                _errmsg += f"{self.metrics.keys()}."
                raise KeyError(_errmsg)
            savename = f"{self.chkpt_prefix}best_{_met_name}_wts.pt"
            savepath = self.chkpts_dir / savename
            _msg = f"`{_met_name}` improved from "
            _msg += f"{self.chkpts_handler[_met_name]:.5f} to "
            _msg += f"{epoch_logs[_met_name]:.5f}. "
            _msg += f"Checkpoint saved @ `{savepath}`."
            if _met_mode == 'min':
                if epoch_logs[_met_name] < self.chkpts_handler[_met_name]:
                    torch.save(net.state_dict(), savepath)
                    self.chkpts_handler[_met_name] = epoch_logs[_met_name]
                    print(_msg)
            else:
                if epoch_logs[_met_name] > self.chkpts_handler[_met_name]:
                    torch.save(net.state_dict(), savepath)
                    self.chkpts_handler[_met_name] = epoch_logs[_met_name]
                    print(_msg)


class TrainingLogger:
    try:
        import wandb
    except ImportError:
        print("[red]wandb is not installed and required for TrainingLogger")
        print("Running `[cyan]pip install wandb[/cyan]` to install...")
        subprocess.run(['pip', 'install', 'wandb'])
        import wandb
        
    def __init__(self,
                 logs_dir: Union[str, Path],
                 config: Optional[dict] = None,
                 run_name: Optional[str] = None,
                 run_id: Optional[str] = None,
                 login_key: Optional[str] = None,
                 job_type: str = 'training',
                 project_name: str = 'MVQAG',
                 save_code: bool = True,
                 keep_n_recent_logs: int = 5) -> None:
        """Training logger abstract class"""
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.keep_n_recent_logs = keep_n_recent_logs
        self.run_name = run_name
        if login_key:
            self.wandb.login(key=login_key, relogin=True, force=True)

        self.run = self.wandb.init(
            job_type=job_type,
            dir=self.logs_dir,
            config=config,
            project=project_name,
            name=run_name,
            save_code=save_code,
            id=run_id
        )

    def log_metrics(self, d: dict) -> None:
        """Log metrics such as loss and accuracy"""
        self.run.log(d)

    def log_hyp(self, d: dict) -> None:
        """Log hyperparameters"""
        return NotImplemented

    def log_model(self, from_dir: Union[str, Path]):
        trained_model_artifact = self.wandb.Artifact('MVQAG', type='model')
        trained_model_artifact.add_dir(from_dir)
        self.run.log_artifact(trained_model_artifact)
        print(f"Trained models uploaded to wandb from `{from_dir}` folder.")

    def log_batch(self, d) -> None:
        """Log a batch of data"""
        return NotImplemented

    def log_predictions(self, d) -> None:
        """Log model's output/predictions"""
        return NotImplemented

    def log_table(self, t: pd.DataFrame) -> None:
        """Log history of training table of every epoch"""
        manage_log_files(logs_dir=self.logs_dir,
                         keep_n_recent_logs=self.keep_n_recent_logs,
                         file_ext='.csv')
        t.to_csv(self.logs_dir/'training_logs.csv')


class VQATrainer:
    def __init__(self,
                 dls: DataLoaders,
                 net: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optm_fn: torch.nn.Module,
                 device: torch.device,
                 metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
                 checkpointer: Optional[Checkpointer] = None,
                 logger: Optional[TrainingLogger] = None,
                 step_lrs: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 epoch_lrs: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 ) -> None:
        """Trainer abstract class

        Args: 
            #TODO
        """

        self.dls, self.net, self.device = dls, net, device
        self.loss_fn, self.optm_fn = loss_fn, optm_fn
        self.metrics, self.logger = metrics, logger
        self.checkpointer = checkpointer
        self.step_lrs = step_lrs
        if isinstance(epoch_lrs, dict):
            self.epoch_lrs = epoch_lrs['lrs']
            self.epoch_lrs_step_arg = epoch_lrs['step']
        else:
            self.epoch_lrs = epoch_lrs

        # Put weights tensor on same device
        if hasattr(self.loss_fn, 'weight') and \
                (self.loss_fn.weight is not None):
            self.loss_fn.weight = self.loss_fn.weight.to(self.device)

        self._bar_fpn = 5  # Display floating point values setting
        self.training_logs_df = None

    def train_one_batch(self, batch):
        target = batch['target'].to(self.device)
        V = batch['inputs']['V'].to(self.device)

        if batch['inputs']['Q'] is None:
            output = self.net(V)
        else:
            Q = {
                k: v.to(self.device) for k, v in batch['inputs']['Q'].items()
            }
            output = self.net(V, Q)

        loss = self.loss_fn(output, target)
        # Backpropagation
        self.optm_fn.zero_grad()
        loss.backward()
        self.optm_fn.step()
        return loss, output, target

    def val_one_batch(self, batch):
        target = batch['target'].to(self.device)
        V = batch['inputs']['V'].to(self.device)
        assert V.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
        Vs_flip = torch.flip(V, [3])

        if batch['inputs']['Q'] is None:
            output = self.net(V)
            output_flip = self.net(Vs_flip)
        else:
            Q = {
                k: v.to(self.device) for k, v in batch['inputs']['Q'].items()
            }
            output = self.net(V, Q)
            output_flip = self.net(Vs_flip, Q)

        # test time augmentation
        output = (output + output_flip) / 2.0

        loss = self.loss_fn(output, target)
        return loss, output, target

    def train_and_val_one_epoch(self,
                                max_train_iters: int = None,
                                max_val_iters: int = None) -> Dict[str, Any]:
        """Train and validate one epoch"""
        # Train
        self.net.train()
        avg_train_loss = []
        for i, batch in track(enumerate(self.dls.trainloader),
                              description=f'Epoch {self.epoch_num}/{self.n_epochs}...',
                              transient=True,
                              total=self.dls.n_train_batches):
            train_loss, output, target = self.train_one_batch(batch)
            if self.step_lrs is not None:
                self.step_lrs.step()
            # print(f"| train_loss: {train_loss:.{self._bar_fpn}f}")
            avg_train_loss.append(train_loss.item())
            if (max_train_iters is not None) and (i >= max_train_iters):
                break
        # Val
        self.net.eval()
        avg_val_loss, val_outputs, val_targets = [], [], []
        with torch.no_grad():
            for i, batch in track(enumerate(self.dls.valloader),
                                  description='validating...',
                                  transient=True,
                                  total=self.dls.n_val_batches):
                val_loss, output, target = self.val_one_batch(batch)
                if isinstance(output, list):
                    val_outputs += output
                    val_targets += target
                else:
                    val_outputs.append(output)
                    val_targets.append(target)
                avg_val_loss.append(val_loss.item())
                if (max_val_iters is not None) and (i >= max_val_iters):
                    break
            # Compute average losses and metrics
            avg_train_loss = torch.tensor(avg_train_loss).mean().item()
            avg_val_loss = torch.tensor(avg_val_loss).mean().item()
            if self.metrics:
                if isinstance(val_outputs[0], torch.Tensor):
                    val_outputs = torch.cat(val_outputs, dim=0).cpu()
                    val_targets = torch.cat(val_targets, dim=0).cpu()
                # Compute metrics
                val_metrics_dict = self.metrics(val_outputs, val_targets)
                _val_metrics_dict = {}
                # Arrange metrics
                for k, _met in val_metrics_dict.items():
                    if isinstance(_met, torch.Tensor):
                        _val_metrics_dict[f'Val{k}'] = round(
                            _met.item(), self._bar_fpn)
                    elif isinstance(_met, dict):
                        _met = {
                            f'Val{k}': round(v.item(), self._bar_fpn)
                            for k, v in _met.items()
                        }
                        _val_metrics_dict.update(_met)
                    else:
                        raise NotImplementedError
                val_metrics_dict = _val_metrics_dict

        epoch_logs = OrderedDict({
            'Epoch': self.epoch_num,
            'TrainLoss': round(avg_train_loss, self._bar_fpn),
            'ValLoss': round(avg_val_loss, self._bar_fpn),
        })
        if self.metrics:
            epoch_logs.update(val_metrics_dict)

        # Epoch LR scheduler
        if self.epoch_lrs:
            if hasattr(self, 'epoch_lrs_step_arg'):
                if self.epoch_lrs_step_arg == 'val_loss':
                    self.epoch_lrs.step(avg_val_loss)
                if self.epoch_lrs_step_arg == 'epoch':
                    self.epoch_lrs.step(self.epoch_num)
                else:
                    _errmsg = f"step={self.epoch_lrs_step_arg} not supported."
                    _errmsg += "Available options: ['val_loss', 'epoch']"
                    raise ValueError(_errmsg)
            else:
                self.epoch_lrs.step()

        return epoch_logs

    def train(self,
              n_epochs: int,
              max_train_iters: int = None,
              max_val_iters: int = None,
              optm_fn: torch.nn.Module = None,
              save_wts: bool = True):
        """Start training"""
        self.n_epochs = n_epochs
        self.max_train_iters = max_train_iters
        self.max_val_iters = max_val_iters
        if optm_fn is not None:
            self.optm_fn = optm_fn
        self.net.to(self.device)
        # Train
        self.interrupted_flag = False
        # Save the best epoch where each metric is minimum
        best_epoch_dict = {}
        try:
            # Epoch number starts from 1
            for self.epoch_num in range(1, n_epochs + 1):
                epoch_logs = self.train_and_val_one_epoch(
                    max_train_iters=max_train_iters,
                    max_val_iters=max_val_iters
                )
                if self.epoch_num == 1:  # starts from 1
                    cols = list(epoch_logs.keys())
                    print('\t'.join(cols))
                    self.training_logs_df = pd.DataFrame(None, columns=cols)
                print('\t'.join(list(map(str, epoch_logs.values()))))

                if self.logger:
                    self.logger.log_metrics(epoch_logs)
                # Save epoch metrics values to a dataframe
                self.training_logs_df = self.training_logs_df.append(
                    epoch_logs, ignore_index=True
                )
                if self.checkpointer and save_wts:
                    # Save checkpoints when metric improves
                    self.checkpointer.save_best_wts(epoch_logs=epoch_logs,
                                                    net=self.net)
                    # Per epoch checkpoint
                    if self.epoch_num % self.checkpointer.chkpt_after == 0:
                        self.checkpointer.create(epoch=self.epoch_num,
                                                 net=self.net,
                                                 loss_fn=self.loss_fn,
                                                 optm_fn=self.optm_fn)
                # Save best epoch information
                if self.epoch_num == 1:
                    for met_name in epoch_logs.keys():
                        if met_name == 'Epoch':
                            continue
                        best_epoch_dict[met_name] = {
                            '@epoch': 1,
                            'value': epoch_logs[met_name]
                        }
                else:
                    for met_name in epoch_logs.keys():
                        if met_name == 'Epoch':
                            continue
                        # For Losses
                        if met_name.endswith('Loss'):
                            if epoch_logs[met_name] < best_epoch_dict[met_name]['value']:
                                best_epoch_dict[met_name] = {
                                    '@epoch': self.epoch_num,
                                    'value': epoch_logs[met_name]
                                }
                        # For accuracy like metrics
                        else:
                            if epoch_logs[met_name] > best_epoch_dict[met_name]['value']:
                                best_epoch_dict[met_name] = {
                                    '@epoch': self.epoch_num,
                                    'value': epoch_logs[met_name]
                                }
                    
        except KeyboardInterrupt:
            self.interrupted_flag = True
            print(f"[red]Interrupted.")
            if self.logger:
                self.logger.run.finish()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.interrupted_flag = True
            print(e)

        finally:
            print(f"\nBest Metrics till epoch {self.epoch_num}:")
            print(best_epoch_dict)
            # detach model from GPU
            # self.net.cpu()
            # Save logs
            if self.logger:
                # Save training history
                if self.training_logs_df is not None:
                    self.logger.log_table(self.training_logs_df)
                # Save model
                if self.checkpointer and (not self.interrupted_flag) and save_wts:
                    self.logger.log_model(
                        from_dir=self.checkpointer.chkpts_dir)
            if self.checkpointer and save_wts:
                # Last epoch checkpoint
                self.checkpointer.create(epoch=self.epoch_num,
                                         net=self.net,
                                         loss_fn=self.loss_fn,
                                         optm_fn=self.optm_fn)
