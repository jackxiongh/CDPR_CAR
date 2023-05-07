import math

def euler2quat(roll, yaw, pitch):
    """
        input: roll, yaw, pitch
    """
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    # 笛卡尔坐标系
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z

def quat2euler(x, y, z, w):
    """
    input: w, x, y, z
    """
    return math.atan2(2 * (y * x + w * z), 1 - 2 * (z * z + y * y))