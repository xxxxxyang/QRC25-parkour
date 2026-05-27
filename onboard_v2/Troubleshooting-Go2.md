# Go2 部署排查步骤

本文用于排查 `sim2real` 中常见问题，例如：

- 行走不稳定
- 遇到障碍无法跨越
- 视觉输入异常
- 策略能跑但动作不对

排查顺序建议固定为：

1. 先确认模型加载正常
2. 再确认视觉链路正常
3. 再确认策略链路正常
4. 最后确认真机控制链路正常

---

## 1. 先检查模型

先确认导出的模型和配置可用：

```bash
cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
python validate_deploy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
```

如果这里不过，先不要继续看视觉或真机控制。

---

## 2. 先查视觉部分

先只跑视觉节点：

```bash
cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
python visual_depth.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
```

重点看下面几项：

- `/forward_depth_image` 是否持续发布
- topic 长度是否等于 `87 * 58 = 5046`
- 深度值是否 finite
- 图像是否明显偏黑、偏白、全零，或者和环境不符

如果这里异常，通常是视觉问题：

- 相机安装位置不对
- 深度裁剪或缩放不对
- 深度范围不对
- 相机输入本身有噪声或遮挡

如果视觉节点本身正常，再看策略。

---

## 3. 可视化检查深度图

可以用 `rviz2` 或 `rqt_image_view` 查看预处理后的深度图。

先运行视觉节点：

```bash
cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
python visual_depth.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced --enable-rgb
```

然后在另一个终端查看 topic：

```bash
ros2 topic list
ros2 topic echo /forward_depth_image --once
```

如果安装了 `rqt_image_view`：

```bash
rqt_image_view
```

选择：

- `/camera/forward_depth`：预处理后的深度图
- `/camera/forward_rgb`：RGB 图像，仅在 `--enable-rgb` 时发布

也可以用 `rviz2` 添加 `Image` 显示：

```bash
rviz2
```

重点看：

- 障碍物是否在深度图中清楚可见
- 机器人前方地面是否连续
- 深度图是否大面积全黑、全白或断裂
- 深度图和 RGB 图像方向是否一致
- 障碍是否被 crop 掉

当前 `visual_depth.py` 默认发布的是预处理后的深度图，不直接发布原始 raw depth。如果要比较 raw depth 和预处理后的差别，可以用 `realsense-viewer` 查看原始深度，或在 `visual_depth.py` 里额外发布 raw depth topic。

---

## 4. 再查策略部分

先保持 dryrun 运行策略：

```bash
cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
python run_depth_policy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
```

建议同时加 `--debug` 看时序：

```bash
python run_depth_policy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced --debug
```

重点看：

- 能否顺利进入 parkour policy
- `L1` 是否能切入策略
- `L2` 是否能退出策略
- `Y` 是否能重置状态
- 是否有 NaN / inf
- action 是否明显抖动

如果 dryrun 就不稳定，通常不是相机问题，而是策略或观测拼接问题。

---

## 5. 区分 vision 问题还是 run 问题

最简单的判断方法是做对照。

### 情况 A：真实深度不稳定，但固定深度正常

说明大概率是视觉部分问题。

可疑项：

- 相机标定或安装位置
- `visual_depth.py` 的 crop / resize
- 深度范围 clip
- `/forward_depth_image` 的数据分布

### 情况 B：真实深度和固定深度都不稳定

说明大概率是 run 部分问题。

可疑项：

- 模型和配置不匹配
- `proprio` 或 `history` 拼接有问题
- `run_depth_policy.py` 的状态切换逻辑
- 真机的 `/lowstate` 或 `/wirelesscontroller` 异常

### 情况 C：dryrun 正常，但真机运动异常

说明大概率是低层控制或机器人状态问题。

可疑项：

- builtin sport service 没有真正关闭
- 电机方向或关节零位不对
- 机器人初始姿态不稳定
- 机械结构、摩擦或电量问题

---

## 6. 对障碍跨越不稳的判断

如果平地还行，但遇到障碍就失败，通常优先怀疑视觉部分：

- 障碍在深度图里不明显
- 相机高度或俯仰角不对
- 深度图被裁掉了关键区域
- 深度范围过窄，远处或近处被截断

如果障碍在视觉图里是清楚的，但动作还是不对，优先怀疑策略部分：

- depth latent 没有正确传入
- `yaw` 修正异常
- `history` 状态不对
- 训练配置和导出配置不一致

---

## 7. 建议的最小排查流程

1. 跑 `validate_deploy.py`
2. 跑 `visual_depth.py`
3. 用 `rqt_image_view` 或 `rviz2` 看 `/camera/forward_depth`
4. 确认 `/forward_depth_image` 长度是 5046
5. 跑 `run_depth_policy.py --debug`
6. 先保持 dryrun
7. 再开 `--nodryrun`

如果前两步都正常，后面仍然不稳，问题大概率在策略或低层控制，不在视觉预处理。
