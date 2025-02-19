import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224
import timm

from models import *

from utils.samplers import DownSampler


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        model.float()

    return model


def load_vit_to_cpu(cfg,wyn_swin_large):
    backbone_name = cfg.backbone

    model = vit_base_patch16_224(pretrained=True).eval()

    model2 = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True)

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp16":
        model.half()
    
    return model,model2


class Trainer:
    def __init__(self, train_dataloader,wyn_swin_large,cfg):
        self.swin_large = wyn_swin_large
        self.num_classes = 52
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))


        #self.classnames =
        d = {}
        maxnum = -1
        lines = open(r"F:/val.txt").readlines()
        for line in lines:
            _,clsname,clsid = line.split(" ")
            clsid = int(clsid)
            if clsid not in d:
                d[clsid] = clsname
            maxnum = max(maxnum,clsid)
        self.classnames = [d[x] for x in range(maxnum)]
        print(self.classnames)


        self.cfg = cfg
        self.train_init_loader = train_dataloader
        self.build_model()
        self._writer = None



    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")

        if cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")

            self.model = PeftModelFromCLIP_context(cfg, self.swin_large, num_classes)

            self.model.to(self.device)
            self.tuner1 = self.model.tuner1
            self.tuner2 = self.model.tuner2
            self.tuner3 = self.model.tuner3
            self.head1 = self.model.head1
            self.head2 = self.model.head2
            self.head3 = self.model.head3


        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            
            torch.cuda.empty_cache()


    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      {"params": self.head.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

        
    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        prompts = self.get_tokenized_prompts(classnames)
        text_features = self.model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[-1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head1.apply_weight(class_means)
        self.head2.apply_weight(class_means)
        self.head3.apply_weight(class_means)



    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)

                if cfg.prec == "amp":
                    with autocast():
                        output = self.model(image)
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                for _c, _y in zip(correct, label):
                    cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                mean_acc = np.mean(np.array(cls_accs))
                many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            output = self.model(image)
            output = output.view(_bsz, _ncrops, -1).mean(dim=1)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)
