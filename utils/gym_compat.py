# utils/gym_compat.py
"""
兼容 Gym / Gymnasium reset/step 的轻量封装：
- reset_env(env, **kwargs) -> (obs, info_dict)
- step_env(env, action) -> (obs, reward, done, info_dict)

无论底层是旧 API(4返回) 还是新 API(5返回)，这里都统一成 4 返回(done 合并 terminated|truncated)。
"""

def reset_env(env, **kwargs):
    """
    兼容:
      Gym(old):   obs
      Gymnasium:  (obs, info)
    统一返回: (obs, info_dict)
    """
    res = env.reset(**kwargs)
    # 新API: 两个返回值
    if isinstance(res, tuple) and len(res) == 2:
        obs, info = res
        return obs, (info if isinstance(info, dict) else {})
    # 旧API: 一个返回值
    return res, {}


def step_env(env, action):
    """
    兼容:
      旧API: obs, reward, done, info
      新API: obs, reward, terminated, truncated, info
    统一返回: (obs, reward, done, info) 其中 done = terminated or truncated
    """
    res = env.step(action)
    if not isinstance(res, tuple):
        # 极端情况：环境自己做了别的封装
        raise TypeError(f"env.step 返回类型异常: {type(res)}")

    n = len(res)
    if n == 5:
        obs, reward, terminated, truncated, info = res
        done = bool(terminated) or bool(truncated)
        info = dict(info) if isinstance(info, dict) else {}
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return obs, reward, done, info
    elif n == 4:
        obs, reward, done, info = res
        info = dict(info) if isinstance(info, dict) else {}
        info["truncated"] = bool(done) and bool(info.get("TimeLimit.truncated", False))
        info["terminated"] = bool(done) and not info["truncated"]
        return obs, reward, bool(done), info
    else:
        raise ValueError(f"env.step 返回数量异常: 期望 4 或 5，收到 {n}: {res}")
