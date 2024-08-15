import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from functools import partial
from einops import rearrange
from utils.utils import instantiate_from_config
import torch.nn.functional as F

from modules.dynamic_modules.utils import draw_dual_grain_256res, draw_dual_grain_256res_color
from models.stage1.utils import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from models.stage2.utils import disabled_train

class Entropy(nn.Sequential):
    def __init__(self, patch_size, image_width, image_height):
        super(Entropy, self).__init__()
        self.width = image_width
        self.height = image_height
        self.psize = patch_size
        # number of patches per image
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        self.hw = int(self.width // self.psize)
        # unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int) -> torch.Tensor:
        """Function that calculates the entropy using marginal probability distribution function of the input tensor
            based on the number of histogram bins.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS].
            sigma: shape [1], gaussian smoothing factor.
            batch: int, size of the batch
        Returns:
            torch.Tensor:
        """
        epsilon = 1e-40
        values = values.unsqueeze(2)
        # 使用 unsqueeze(2) 在 values 张量的第二个维度上插入一个新维度。结果是将 values 张量的形状从 [BxNx1] 变为 [BxNx1x1]。
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
        # 使用高斯函数计算每个残差的概率密度值。这个操作应用于每个块和每个分箱。
        pdf = torch.mean(kernel_values, dim=1)
        # 对所有块的高斯核值取平均，得到边缘概率密度函数（PDF）
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        # 计算并应用归一化，使得 PDF 的总和为 1。这里添加了 epsilon 以防止出现除零错误。
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        # 根据公式  S = -\sum p(x) \log p(x) ，计算每个块的熵值。
        entropy = entropy.reshape((batch, -1))
        entropy = rearrange(entropy, "B (H W) -> B H W", H=self.hw, W=self.hw)
        # 首先将熵张量重新整形为 [batch, -1]，然后使用 rearrange 函数将其重新排列为 [B, H, W] 的形状，这样每个批次的熵图可以按照原图像的块布局进行排列
        return entropy

    def forward(self, inputs: Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]
        # 将彩色图像转换为灰度图像。转换公式基于加权平均法，常用于图像处理。得到的 gray_images 形状为 [batch_size, 1, height, width]，其中每个图像只有一个通道（灰度值）。
        #系数是一个标准，常用于图像预处理，图像压缩
        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)
        # reshape to (batch * h*w/ (patch_size*patch_size) x (patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        # 转置操作，将张量的维度交换，形状变为 [batch_size, num_patches, patch_size*patch_size]。这使得每个块成为一个独立的元素。
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
                                        (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(-1, 1, 32).to(device=inputs.device),
                               sigma=torch.tensor(0.01), batch=batch_size)

        return entropy

class DualGrainVQModel(pl.LightningModule):
    def __init__(self,
                 encoderconfig,
                 decoderconfig,
                 lossconfig,
                 vqconfig,

                 quant_before_dim,
                 quant_after_dim,
                 quant_sample_temperature = 0., 
                 ckpt_path = None,
                 ignore_keys = [],
                 image_key = "image",
                 monitor = None,
                 warmup_epochs = 0,
                 loss_with_epoch = True,
                 scheduler_type = "linear-warmup_cosine-decay",
                 entropy_patch_size = 16, # maximum patch size of all granularity
                 image_size = 256,
                 ):
        super().__init__()
        '''
        encoderconfig, decoderconfig, lossconfig, vqconfig:
	•	这些配置参数用于实例化编码器、解码器、损失函数和量化模块。通常，这些是配置字典或包含模型结构和超参数的配置文件。
    	•	quant_before_dim, quant_after_dim:
	•	这些参数定义了量化模块前后张量的通道维度。在量化过程中，张量的通道维度可能会发生变化，这些参数用于定义这种变化。
	•	quant_sample_temperature:
	•	量化采样的温度参数，通常用于控制在采样过程中引入的随机性。
	•	ckpt_path:
	•	用于指定模型检查点的路径，如果提供了这个路径，模型将从该检查点加载参数。
	•	ignore_keys:
	•	这是一个列表，包含在加载检查点时需要忽略的状态字典键。
	•	image_key:
	•	定义图像数据在输入数据字典中的键。
	•	monitor:
	•	用于监控训练过程中的某个指标，以便进行模型选择或学习率调度。
	•	warmup_epochs, loss_with_epoch, scheduler_type:
	•	这些参数用于设置训练调度器（scheduler），例如学习率的预热和衰减策略。
	•	entropy_patch_size, image_size:
	•	entropy_patch_size 定义了计算熵时的最大图像块大小（patch size）。
	•	image_size 是输入图像的尺寸。

        '''
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoderconfig)
        self.decoder = instantiate_from_config(decoderconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.quantize = instantiate_from_config(vqconfig)

        self.quant_conv = torch.nn.Conv2d(quant_before_dim, quant_after_dim, 1)
        # •	定义了一个 2D 卷积层，用于将特征图从 quant_before_dim 转换为 quant_after_dim。卷积核大小为 1x1，这意味着它只会改变通道数，不会改变空间维度。
		#用途:这个卷积层通常用于调整特征图的通道数，使其与量化模块的输入要求匹配。
        self.post_quant_conv = torch.nn.Conv2d(quant_after_dim, quant_before_dim, 1)
        # 这个卷积层用于将量化后的特征表示映射回原始的特征空间，以便后续的解码操作
        self.quant_sample_temperature = quant_sample_temperature
        # 设置量化采样的温度参数，这个参数通常用于控制采样过程中的随机性。温度越高，采样越随机；温度越低，采样越确定。
        self.entropy_patch_size = entropy_patch_size
        self.image_size = image_size 
        self.entropy_calculation = Entropy(entropy_patch_size, image_size, image_size)
        self.entropy_calculation = self.entropy_calculation.eval()
        #将熵计算模块设置为评估模式（evaluation mode），这意味着在计算熵时，模块中的所有参数将被固定，且不会发生梯度计算。
        self.entropy_calculation.train = disabled_train

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        # 如果提供了检查点路径（ckpt_path），模型将从指定路径加载预训练参数。ignore_keys 用于忽略特定的参数键，这些键不会从检查点加载。
        if monitor is not None:
            self.monitor = monitor
        # 设置图像的键值（image_key）和监控指标（monitor）。监控指标通常用于在训练过程中跟踪模型性能，并在早停（early stopping）或模型保存时使用。
        self.warmup_epochs = warmup_epochs
        self.loss_with_epoch = loss_with_epoch
        self.scheduler_type = scheduler_type

    def init_from_ckpt(self, path, ignore_keys=list()):
        #这个方法用于从指定的检查点路径加载模型参数。它会遍历状态字典中的键，并删除那些在 ignore_keys 列表中的键，然后将剩余的键加载到模型中。
        sd = torch.load(path, map_location="cpu")["state_dict"]
        # 使用 torch.load 加载检查点文件，并提取其中的 state_dict。
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    # 检查 k 是否以 ik 开头，如果是的话，就意味着这个键应该被忽略。 
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # 它执行的步骤包括计算输入图像的熵，使用编码器生成特征表示，然后将这些特征通过量化模块进行量化。
        x_entropy = self.entropy_calculation(x)
        h_dict = self.encoder(x, x_entropy)
        h = h_dict["h_dual"]
        grain_indices = h_dict["indices"]
        codebook_mask = h_dict["codebook_mask"]
        gate = h_dict["gate"]

        h = self.quant_conv(h) # 对编码后的特征进行一次卷积操作，通常是为了调整特征的维度，
        quant, emb_loss, info = self.quantize(x=h, temp=self.quant_sample_temperature, codebook_mask=codebook_mask)
        # 使用量化模块 quantize 对特征进行量化。量化操作将连续特征映射为离散代码，返回量化后的特征 quant，嵌入损失 emb_loss，以及其他附加信息 info。
        return quant, emb_loss, info, grain_indices, gate, x_entropy

    def decode(self, quant, grain_indices=None):
        quant = self.post_quant_conv(quant)
        # 	量化后的特征 quant 首先通过一个 1x1 卷积层 post_quant_conv。这个卷积层的作用通常是将特征映射回原始的通道数
        dec = self.decoder(quant, grain_indices)
        # grain_indices 是在编码过程中生成的粒度索引
        return dec

    def forward(self, input):
        quant, diff, _, grain_indices, gate, x_entropy = self.encode(input)
        
        dec = self.decode(quant, grain_indices)
        return dec, diff, grain_indices, gate, x_entropy

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
            # 如果输入 x 的形状只有 3 个维度（即没有通道维度），则在最后一个维度添加一个新的维度，使其变为 4 维张量 [batch_size, height, width, channels]。
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
            # 如果输入的通道数（即第二个维度）不是 3，则将张量的维度重新排列为 [batch_size, channels, height, width] 的形式，这是大多数卷积神经网络所期望的格式。
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices, gate, x_entropy = self(x)
        # 调用模型的 forward 方法，得到重构图像 xrec、量化损失 qloss、粒度索引 indices、门控信息 gate 和输入的熵 x_entropy
        ratio = indices.sum() / (indices.size(0) * indices.size(1) * indices.size(2))
        # ！！ 计算粒度索引中活跃元素的比例（fine-grained 比例），用于评估模型在不同粒度上的表现。

        if optimizer_idx == 0:
            # autoencode
            if self.loss_with_epoch:
                # 计算自编码器的损失。如果 loss_with_epoch 为 True，损失会基于当前的 epoch 计算，否则基于全局步数计算。
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, last_layer=self.get_last_layer(), split="train", gate=gate)
            else:
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train", gate=gate)
            
            self.log("train_aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("train_fine_ratio", ratio, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            rec_loss = log_dict_ae["train_rec_loss"]
            self.log("train_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            del log_dict_ae["train_rec_loss"]
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            # 	记录自编码器损失、重构损失和细粒度比例等信息。
            return aeloss

        if optimizer_idx == 1:
            # discriminator 判别器
            if self.loss_with_epoch:
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.current_epoch, last_layer=self.get_last_layer(), split="train")
            else:
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")

            self.log("train_discloss", discloss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        # 定义了在验证过程中每个批次的操作。它与 training_step 类似，但用于评估模型在验证集上的表现
        x = self.get_input(batch, self.image_key)
        xrec, qloss, indices, gate, x_entropy = self(x)
        ratio = indices.sum() / (indices.size(0) * indices.size(1) * indices.size(2))
        self.log("val_fine_ratio", ratio, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.loss_with_epoch:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.current_epoch, last_layer=self.get_last_layer(), split="val", gate=gate)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.current_epoch, last_layer=self.get_last_layer(), split="val")
        else:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val", gate=gate)
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val_rec_loss"]
        self.log("val_rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val_rec_loss"]
        self.log("val_aeloss", aeloss, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict
    
    def configure_optimizers(self):
        # 定义了 configure_optimizers 方法，它负责配置优化器和学习率调度器。在 PyTorch Lightning 中，configure_optimizers 是一个关键方法，
        # 用于指定模型的优化策略，包括哪些参数需要优化、使用哪种优化算法，以及如何调整学习率。
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        # 一个用于自编码器（encoder 和 decoder）                          
            # Adam 优化器的 betas 参数设为 (0.5, 0.9)，这通常是用于生成模型（如 GANs）的标准设置。                      
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        # 用于判别器（discriminator）。
        warmup_steps = self.steps_per_epoch * self.warmup_epochs
        # 学习率调度策略：计算出预热的步数，通常是 steps_per_epoch 乘以预热的 epoch 数量
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
            }
            # 	LambdaLR 是 PyTorch 中的一个学习率调度器。它允许你根据一个自定义的 lambda 函数来调整学习率。
            #	在这个例子中，LambdaLR 的第一个参数是 opt_ae，这是优化器对象。
            #	第二个参数是 Scheduler_LinearWarmup(warmup_steps)，这是一个自定义的 lambda 函数或类，定义了如何根据训练步数调整学习率。
            #	Scheduler_LinearWarmup(warmup_steps):
            #	Scheduler_LinearWarmup 是一个自定义的调度器类或函数，专门用于实现线性学习率预热。warmup_steps 是一个整数，表示学习率预热的步数。在这些步数内，学习率会线性增加到设定的最大值。
            # interval:interval 指定了调度器更新学习率的频率。   "step" 表示调度器在每个训练步骤（step）后更新一次学习率。
            # frequency:frequency 指定了调度器在指定 interval 内执行的频率。这里的 1 表示每次步数更新时都要调整学习率。
            # 例如，如果 interval 是 "epoch" 且 frequency 是 1，则调度器会在每个 epoch 结束后更新学习率。如果 frequency 是 2，则每隔两个 epoch 更新一次学习率。
            #这意味着每经过一个 batch 的训练，学习率都会根据当前的步数和 Scheduler_LinearWarmup 的函数进行调整。
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
            }
        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
            }
            scheduler_disc = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=self.training_steps, multipler_min=multipler_min)), "interval": "step", "frequency": 1,
            }
        else:
            raise NotImplementedError()

        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]

    def get_last_layer(self):
        '''
        这个方法用于获取模型解码器的最后一层权重，通常在计算损失时会用到最后一层的权重。
        这段代码首先尝试获取解码器的 conv_out 层的权重。如果 conv_out 层存在，它的权重将被返回。
	•	如果 conv_out 层不存在，代码会捕获异常并返回 self.decoder.last_layer。这个字段可能是解码器中的另一层或是预备的最后一层。
        '''
        try:
            return self.decoder.conv_out.weight
        except:
            return self.decoder.last_layer

    def log_images(self, batch, **kwargs):
        '''
        获取输入图像:
	•	x = self.get_input(batch, self.image_key) 获取输入图像 x，并将其移动到模型所在的设备上（例如 GPU）。
	•	前向传播:
	•	xrec, _, grain_indices, gate, x_entropy = self(x) 通过模型的前向传播获取重构图像、粒度索引和熵图。
	•	记录图像信息:
	•	log["inputs"] 存储原始输入图像。
	•	log["reconstructions"] 存储重构后的图像。
	•	log["grain_map"] 存储粒度图（通过 draw_dual_grain_256res_color 方法生成）。
	•	log["entropy_map"] 存储熵图，这里先对熵值进行了归一化，再绘制成图像。
	•	返回值:
	•	返回一个包含输入、重构图像、粒度图和熵图的字典，用于记录或可视化。
    '''
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, grain_indices, gate, x_entropy = self(x)

        log["inputs"] = x
        log["reconstructions"] = xrec
        # log["grain"] = draw_dual_grain_256res(images=x.clone(), indices=grain_indices)
        log["grain_map"] = draw_dual_grain_256res_color(images=x.clone(), indices=grain_indices, scaler=0.7)
        x_entropy = x_entropy.sub(x_entropy.min()).div(max(x_entropy.max() - x_entropy.min(), 1e-5))
        log["entropy_map"] = draw_dual_grain_256res_color(images=x.clone(), indices=x_entropy, scaler=0.7)
        return log
    
    def get_code_emb_with_depth(self, code):
        embed = self.quantize.get_codebook_entry(code)
        # embed = rearrange(embed, "b h w c -> b c h w")
        return embed
        # return self.quantize.embed_code_with_depth(code)
