import logging
import time

from vidgen.engine.defaults import DefaultTrainer
from vidgen.utils import comm
from vidgen.utils.logger import setup_logger


class Trainer(DefaultTrainer):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.

    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        logger = logging.getLogger("vidgen")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        # Assume these objects must be constructed in this order.
        self.model = self.build_model(cfg)
        self.optimizers, self.checkpointers = self.model.configure_optimizers_and_checkpointers()
        self.data_loader, dataset_len = self.build_train_loader(cfg)
        self._data_loader_iter = iter(self.data_loader)
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            self.model.wrap_parallel(device_ids=[comm.get_local_rank()], broadcast_buffers=False)
        super().__init__(cfg)
        self.model.train()
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.gan_mode_on = cfg.GAN_MODE_ON
        self.supervised_max_iter = cfg.SOLVER.SUPERVISED_MAX_ITER
        self.d_update_ratio = cfg.SOLVER.D_UPDATE_RATIO
        self.d_init_iters = cfg.SOLVER.D_INIT_ITERS
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        self.accumulation_steps = cfg.SOLVER.ACCUMULATION_STEPS

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        g_loss_dict = {}
        d_loss_dict = {}

        if self.gan_mode_on:
            self.model.set_generator_requires_grad(True)
            self.model.set_discriminator_requires_grad(False)

        if not self.gan_mode_on or self.iter < self.supervised_max_iter:
            g_loss_dict = self.model(data, mode='supervised')
            g_losses = sum(g_loss_dict.values())
            self._detect_anomaly(g_losses, g_loss_dict)
            g_losses.backward()
            if (self.iter + 1) % self.accumulation_steps == 0:
                for item in self.optimizers:
                    item["optimizer"].step()
                for item in self.optimizers:
                    item["optimizer"].zero_grad()
        else:
            """
            Run generator one step
            """
            assert self.gan_mode_on
            self.model.set_generator_requires_grad(True)
            self.model.set_discriminator_requires_grad(False)
            if self.iter % self.d_update_ratio == 0 and self.iter > self.d_init_iters:
                for item in self.optimizers:
                    if item["type"] == "generator":
                        item["optimizer"].zero_grad()
                g_loss_dict = self.model(data, mode='generator')
                g_losses = sum(g_loss_dict.values())
                self._detect_anomaly(g_losses, g_loss_dict)
                g_losses.backward()
                for item in self.optimizers:
                    if item["type"] == "generator":
                        item["optimizer"].step()

            """
            Run discriminator one step
            """
            self.model.set_generator_requires_grad(False)
            self.model.set_discriminator_requires_grad(True)
            for item in self.optimizers:
                if item["type"] == "discriminator":
                    item["optimizer"].zero_grad()
            d_loss_dict = self.model(data, mode='discriminator')
            d_losses = sum(d_loss_dict.values())
            self._detect_anomaly(d_losses, d_loss_dict)
            d_losses.backward()
            for item in self.optimizers:
                if item["type"] == "discriminator":
                    item["optimizer"].step()

        """
        Write metrics
        """
        metrics_dict = {**g_loss_dict, **d_loss_dict}
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
