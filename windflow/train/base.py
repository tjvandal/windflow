import wandb
from pytorch_lightning import LightningModule

import torch
import torchvision


def scale_image(x):
    xmn = torch.min(x)
    xmx = torch.max(x)
    return (x - xmn) / (xmx - xmn)


class BaseTrainer(LightningModule):
    def __init__(
        self,
        # model_name,
        # model_path,
        lr=1e-4,
    ):
        super().__init__()
        # self.model_name = model_name
        # self.model_path = model_path
        self.lr = lr
        # self.global_step = 0
        # self.wandb_run = wandb.init(project='windflow')

    def configure_optimizers(self):
        # set optimizer
        params_list = list(self.parameters())
        optimizer = torch.optim.Adam(params_list, lr=self.lr, weight_decay=1e-4)
        return optimizer

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_trainer(self):
        try:
            return getattr(self, "trainer", None)
        except RuntimeError:  # not attached to a trainer
            return None

    @property
    def mode(self):
        trainer = self.get_trainer()
        if trainer is None:
            return None
        if trainer.sanity_checking:
            return None
        if trainer.testing:
            return "test"
        if trainer.predicting:
            return "predict"
        if trainer.evaluating:
            return "eval"
        if trainer.training:
            return "train"
        return None  # inference ?

    def log_scalar(self, x, name):
        # tfwriter = self.get_tfwriter(train)
        # tfwriter.add_scalar(name, x, self.global_step)
        logger = self.logger.experiment
        prefix = self.mode
        logger.log({f"{prefix}/{name}": x})

    def log_image_grid(self, img, name, N=4):
        """
        img of shape (N, C, H, W)
        """
        # tfwriter = self.get_tfwriter(train)
        logger = self.logger.experiment
        prefix = self.mode
        img_grid = torchvision.utils.make_grid(img[:N])
        logimg = wandb.Image(img_grid)
        logger.log({f"{prefix}/{name}": logimg})

    def log_flow_grid(self, flows, name, N=4):
        prefix = self.mode
        logger = self.logger.experiment
        U_grid = torchvision.utils.make_grid(flows[:N, :1])
        V_grid = torchvision.utils.make_grid(flows[:N, 1:])
        intensity = (U_grid**2 + V_grid**2) ** 0.5
        # tfwriter.add_image(f'{name}/U', scale_image(U_grid), self.global_step)
        # tfwriter.add_image(f'{name}/V', scale_image(V_grid), self.global_step)
        # tfwriter.add_image(f'{name}/intensity', scale_image(intensity), self.global_step)
        logs = {}
        logs[f"{prefix}/{name}/U"] = wandb.Image(U_grid)
        logs[f"{prefix}/{name}/V"] = wandb.Image(V_grid)
        logs[f"{prefix}/{name}/Intensity"] = wandb.Image(intensity)
        logger.log(logs)

    def step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx)
