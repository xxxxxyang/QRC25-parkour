# 在真实 Unitree Go2 上部署模型

本文说明如何在真实 Unitree Go2 机器人上部署 `onboard_v2` 策略。

相比原来的 `onboard` 文件夹，当前版本使用：

- `visual_depth.py` 进行深度图预处理，并发布 `/forward_depth_image`
- `run_depth_policy.py` 进行策略推理和低层控制
- `vision_policy.py` 加载导出的 JIT 策略和 depth encoder

---

## 在 Go2 上安装依赖

1. 确认 Jetson 系统和 JetPack 已经配置完成。

2. 安装 ROS2 Foxy/Humble 和 Unitree ROS2 环境。

3. 创建用于部署的 Python 环境。

   ```bash
   sudo apt-get install python3-pip python3-dev python3-venv
   python3 -m venv onboard_venv
   source onboard_venv/bin/activate
   ```

4. 安装适用于 Jetson 平台的 PyTorch。

   请根据设备选择对应的 aarch64 wheel。

5. 安装 `ros2_numpy`。

   ```bash
   pip install transformations pybase64
   mkdir -p ros2_numpy_ws/src
   cd ros2_numpy_ws/src
   git clone https://github.com/nitesh-subedi/ros2_numpy.git
   cd ../
   colcon build
   ```

6. 将本项目拷贝到 onboard 机器上。

   至少需要保留以下内容：

   - `onboard_v2`
   - `rsl_rl`
   - 导出的模型日志目录，包括 `traced/config.json`、`base_jit.pt`、`vision_weight.pt`

7. 安装 `rsl_rl`。

   ```bash
   pip install -e ./rsl_rl
   ```

---

## 在 Go2 上运行模型

***免责声明：*** 机器人运动时请始终使用安全绳。

1. 将机器人放在地面上并开机。

2. 关闭内置 sport service。

3. 确认 Intel RealSense D435i 安装在标定时的位置。

4. 在机器人上打开两个终端，并在两个终端中 source 相同环境：

   ```bash
   source /opt/ros/foxy/setup.bash
   source ~/ros2_numpy_ws/install/setup.bash
   source ~/onboard_venv/bin/activate
   ```

5. 在终端 `T_visual` 中运行：

   ```bash
   cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
   python visual_depth.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
   ```

6. 在终端 `T_run` 中运行：

   ```bash
   cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
   python run_depth_policy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
   ```

7. 默认是 dryrun 模式。

   dryrun 模式下，程序不会真正向电机发送动作。建议先用该模式检查。

8. 如果需要让机器人实际运动，添加 `--nodryrun`。

   ```bash
   python run_depth_policy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced --nodryrun
   ```

---

## 按键行为

- `L1`：退出 sport mode，进入 parkour policy
- `L2`：退出 parkour policy，回到 sport mode
- `Y`：重置 policy 状态

策略使用：

- `/forward_depth_image` 作为深度输入
- `/lowstate` 获取 proprioception
- `/wirelesscontroller` 进行模式切换
- `/lowcmd` 发布低层动作

---

## 注意事项

1. `visual_depth.py` 会发布预处理后的深度图和 `/forward_depth_image`。

2. `run_depth_policy.py` 会加载：

   - `base_jit.pt`
   - `vision_weight.pt`

   并重建 depth encoder 和 locomotion policy wrapper。

3. 在真机运行前，可以先检查导出的模型：

   ```bash
   python validate_deploy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
   ```

