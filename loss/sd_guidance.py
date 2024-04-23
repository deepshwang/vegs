import torch
import torch.nn as nn
import diffusers
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
import os

class LoRADiffusionRegularizer(nn.Module):
    """
    Regularizing augmented views with SD model fine-tuned with LoRA
    """

    def __init__(self, args, kitti_args, sd_args, max_iters):
        super().__init__()
        self.dtype = torch.float32
        self.args = sd_args
        if args.data_type == "kitti":
            subdir = kitti_args.seq
        elif args.data_type == "kitti360":
            subdir = os.path.join(kitti_args.seq, f"{str(kitti_args.start_frame).zfill(10)}_{str(kitti_args.end_frame).zfill(10)}")
        self.model_path = os.path.join(sd_args.lora_model_dir, args.data_type, subdir)
        if sd_args.lora_checkpoint_iter is not None:
            self.model_path = os.path.join(self.model_path, f"checkpoint-{sd_args.lora_checkpoint_iter}")
        self.sd_model_key = sd_args.sd_model_key
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(self.sd_model_key, torch_dtype=self.dtype)
        pipe.unet.load_attn_procs(self.model_path)
        pipe = pipe.to("cuda")

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(self.sd_model_key, subfolder="scheduler", torch_dtype=self.dtype)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to("cuda")

        self.text_embeddings = self.prepare_text_embeds()
        del self.tokenizer, self.text_encoder
        torch.cuda.empty_cache()

        self.min_step=sd_args.sd_min_step
        self.max_step=sd_args.sd_max_step
        self.guidance_scale=sd_args.sd_guidance_scale

        self.start_iter = sd_args.start_guiding_from_iter
        self.max_iters = max_iters

        print(f"Loaded SD model base {self.sd_model_key} finetuned with LoRA at {self.model_path}")
    
    def forward(self, pred_rgb, iter):
        latents = self.encode_imgs(pred_rgb)
        max_step = int(self.max_step * (1 - (iter - self.start_iter) / (self.max_iters - self.start_iter)))
        t = torch.randint(self.min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device="cuda")

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=self.text_embeddings).sample

            # perform guidance (high scale from paper, but 7.5 works the best, which is the guidance scale used for training LoRA)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_pos - noise_pred_uncond)


        # [1] Score-matching loss function (diffusionerf)
        if self.args.guidance_mode == "score-matching":
            sigma_lambda = (1. - self.alphas_cumprod[t]).sqrt()
            assert sigma_lambda > 0.

            grad_log_prior_prob = -noise_pred.detach() * (1. / sigma_lambda)
            diffusion_pseudo_loss = -torch.sum(self.args.sm_lambda * grad_log_prior_prob * latents)
            return diffusion_pseudo_loss
        
        # [2] SDS loss function (stable-dreamfusion)
        elif self.args.guidance_mode == "sds":
            w = 1 - self.alphas_cumprod[t]
            grad =  self.args.sds_grad_scale * w[:, None, None, None] * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            targets = (latents- grad).detach()
            loss = 0.5 * torch.nn.functional.mse_loss(latents.float(), targets, reduction="sum") / latents.shape[0]
            return loss

        else:
            raise NotImplementedError(f"Unknown diffusion regularization method {self.args.mode}")

    def prepare_text_embeds(self):
        pos_embeds = self.get_text_embeds(self.args.prompts)
        neg_embeds = self.get_text_embeds(self.args.negative_prompts)
        text_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
        return text_embeds

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to("cuda"))[0]
        return embeddings
