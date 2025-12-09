import inputs

# 屏蔽 LED 解析
inputs.DeviceManager._find_leds = lambda self: None

# 初始化 DeviceManager
dm = inputs.DeviceManager()
print("DeviceManager initialized successfully.")

# 打印 type_codes（可选）
#print(dm.codes['type_codes'].keys())

# 检测是否有手柄连接
if dm.gamepads:
    print(f"Detected {len(dm.gamepads)} gamepad(s):")
    for gp in dm.gamepads:
        print(f"- {gp.name}")
else:
    print("No gamepad detected.")
