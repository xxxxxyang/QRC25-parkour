# Deploy the model on your real Unitree Go2 robot

This file shows how to deploy the `onboard_v2` policy on a real Unitree Go2 robot.

Compared with the original `onboard` folder, this version uses:

- `visual_depth.py` for depth preprocessing and `/forward_depth_image` publishing
- `run_depth_policy.py` for policy inference and low-level control
- `vision_policy.py` for loading the exported JIT policy and depth encoder

---

## Install dependencies on Go2

1. Make sure your Jetson system and JetPack are ready.

2. Install ROS2 Foxy/Humble and the Unitree ROS2 environment.

3. Create a Python environment for deployment.

   ```bash
   sudo apt-get install python3-pip python3-dev python3-venv
   python3 -m venv onboard_venv
   source onboard_venv/bin/activate
   ```

4. Install PyTorch for the Jetson platform.

   Use the appropriate aarch64 wheel for your device.

5. Install `ros2_numpy`.

   ```bash
   pip install transformations pybase64
   mkdir -p ros2_numpy_ws/src
   cd ros2_numpy_ws/src
   git clone https://github.com/nitesh-subedi/ros2_numpy.git
   cd ../
   colcon build
   ```

6. Copy this project to the onboard machine.

   At minimum, keep these folders together:

   - `onboard_v2`
   - `rsl_rl`
   - the exported model log directory, including `traced/config.json`, `base_jit.pt`, `vision_weight.pt`

7. Install `rsl_rl`.

   ```bash
   pip install -e ./rsl_rl
   ```

---

## Run the model on Go2

***Disclaimer:*** Always use a safety belt when the robot moves.

1. Put the robot on the ground and power it on.

2. Turn off the builtin sport service.

3. Make sure the Intel RealSense D435i is mounted at the calibrated position.

4. Open two terminals on the robot and source the same environment in both:

   ```bash
   source /opt/ros/foxy/setup.bash
   source ~/ros2_numpy_ws/install/setup.bash
   source ~/onboard_venv/bin/activate
   ```

5. In terminal `T_visual`, run:

   ```bash
   cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
   python visual_depth.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
   ```

6. In terminal `T_run`, run:

   ```bash
   cd /home/xyang/parkour/QRC25-parkour/onboard_v2/go2
   python run_depth_policy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
   ```

7. The default mode is dryrun.

   In dryrun mode, the robot will not actually move its motors. This is the recommended first check.

8. If you want to let the robot move, add `--nodryrun`.

   ```bash
   python run_depth_policy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced --nodryrun
   ```

---

## Behavior

- `L1`: exit sport mode and enter parkour policy
- `L2`: exit parkour policy and return to sport mode
- `Y`: reset policy state

The policy uses:

- `/forward_depth_image` as input depth
- `/lowstate` for proprioception
- `/wirelesscontroller` for mode switching
- `/lowcmd` for low-level action publishing

---

## Notes

1. `visual_depth.py` publishes the preprocessed depth image and `/forward_depth_image`.

2. `run_depth_policy.py` loads:

   - `base_jit.pt`
   - `vision_weight.pt`

   and reconstructs the depth encoder + locomotion policy wrapper.

3. Before running on the robot, you can first verify the exported model by:

   ```bash
   python validate_deploy.py --logdir /home/xyang/parkour/QRC25-parkour/legged_gym/logs/go2-onboard/traced
   ```

