import torch
import torch.nn.functional as F
import random


class DepthNoiseManager:
    """
    管理深度图噪声：
      1) 依据 episode 选择 mapping condition
      2) 根据 curriculum level 动态调节噪声强度
      3) 按顺序添加：delay → 环境噪声 → 高斯测距噪声 + 噪点噪声
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # 记录当前 episode 所选择的 mapping condition
        self.mapping_condition = None


    # Episode 开头选择 mapping condition
    def sample_mapping_condition(self):
        p = random.random()
        if p < 0.6:
            self.mapping_condition = "nominal"     # 60%
        elif p < 0.6 + 0.3:
            self.mapping_condition = "offset"      # 30%
        else:
            self.mapping_condition = "noisy"       # 10%


    # 2. curriculum 控制噪声强度（可自行定规则）
    def compute_intensity(self, csk):
        """
        csk: curriculum stage/iteration
        返回一个 noise_scale（乘子）
        """
        # 你未来可以替换成 performance-based curriculum
        base = {
            "nominal": 0.3,
            "offset": 0.7,
            "noisy": 1.2,
        }[self.mapping_condition]

        # 简单线性 curriculum 示例：越后期噪声越强
        curriculum_scale = min(1.0 + 0.0001 * csk, 2.0)

        return base * curriculum_scale

    # delay 噪声
    def apply_delay(self, depth_image, buffer, noise_scale):

        if self.mapping_condition != "offset":
            return depth_image

        buf_len = buffer.shape[0]

        # 1) 根据噪声强度决定触发概率
        p_delay = min(1.0, 0.3 + noise_scale * 0.5)
        if random.random() > p_delay:
            return depth_image

        # 2) 根据噪声大小决定最大延迟帧
        max_delay = int(1 + noise_scale * 2)
        max_delay = min(max_delay, buf_len - 1)

        s_t = random.randint(0, max_delay)
        delayed = buffer[-1 - s_t]

        # 3) 噪声越大越偏向 delayed
        w = min(1.0, 0.3 + noise_scale * 0.7)

        # 邻帧插值平滑
        x = torch.rand(1, device=self.device).item()
        blended = (1 - x) * depth_image + x * buffer[-2]

        return (1 - w) * blended + w * delayed


    # 环境噪声（patch degradation + jitter）
    def apply_environment_noise(self, depth_image, noise_scale):
        """
        mapping condition = noisy（10% 档） → 添加环境噪声
        patch 以小概率变为全 0 或全最大深度（模拟反光/穿透/丢失）
        """
        if self.mapping_condition != "noisy":
            return depth_image

        H, W = depth_image.shape
        noisy_img = depth_image.clone()

        K = random.randint(3, 7)
        min_size = self.cfg.depth.env_patch_min
        max_size = self.cfg.depth.env_patch_max

        # 小概率触发 patch 置零/置最大
        p_spoof = getattr(self.cfg.depth, "env_spoof_prob", 0.15)

        for _ in range(K):
            ph = random.randint(min_size, max_size)
            pw = random.randint(min_size, max_size)
            y = random.randint(0, max(1, H - ph))
            x = random.randint(0, max(1, W - pw))

            patch = noisy_img[y:y + ph, x:x + pw]

            # 小概率把 patch 变成全 0 或全最大深度
            if random.random() < p_spoof:
                if random.random() < 0.5:
                    # patch 全为 0 → 透光、激光穿透、深度缺失
                    noisy_img[y:y + ph, x:x + pw] = torch.zeros_like(patch)
                else:
                    # patch 全为 far_clip → 反光 saturate
                    noisy_img[y:y + ph, x:x + pw] = torch.ones_like(patch) * self.cfg.depth.far_clip
                continue
            # 正常 patch 像素扰动（乘 α）
            alpha = random.uniform(0.6, 1.4) * noise_scale
            noisy_img[y:y + ph, x:x + pw] = patch * alpha

        # 随机 jitter
        jitter = torch.randn_like(noisy_img) * 0.01 * noise_scale
        return noisy_img + jitter

    # 高斯测距噪声 + 椒盐噪点
    def apply_gaussian_and_saltpepper(self, depth_image, noise_scale):
        # 测距噪声：N(0, sigma*depth)
        sigma = self.cfg.depth.range_noise_sigma * noise_scale
        gaussian = torch.randn_like(depth_image) * sigma * torch.abs(depth_image)
        depth_noisy = depth_image + gaussian

        # 椒盐噪声
        sp_prob = self.cfg.depth.outlier_prob * noise_scale
        mask = torch.rand_like(depth_image) < sp_prob
        # 椒盐值：随机最远/最近
        sp_val = torch.where(torch.rand_like(depth_image) > 0.5,
                             torch.tensor(self.cfg.depth.far_clip, device=self.device),
                             torch.tensor(self.cfg.depth.near_clip, device=self.device))
        depth_noisy = torch.where(mask, sp_val, depth_noisy)
        return depth_noisy

    # 总入口：对深度图添加所有噪声
    def add_noise(self, depth_image, csk):
        """
        流程：
        1) delay（仅 offset）
        2) environment noise（仅 noisy）
        3) Gaussian + salt & pepper
        """
        noise_scale = self.compute_intensity(csk)

        depth_image = self.apply_delay(depth_image, noise_scale)
        depth_image = self.apply_environment_noise(depth_image, noise_scale)
        depth_image = self.apply_gaussian_and_saltpepper(depth_image, noise_scale)

        return depth_image
