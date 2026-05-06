"""
注册表模式最小示例 —— 对应 PreTrainedConfig + register_subclass 的核心机制
"""

from dataclasses import dataclass
from typing import ClassVar, Dict


# ── 第一步：基类，维护一张 名字→类 的注册表 ──────────────────────────────────

class Registry:
    _registry: ClassVar[Dict[str, type]] = {}   # 所有子类共享这一张表

    @classmethod
    def register(cls, name: str):
        """当装饰器用：@Registry.register("act")"""
        def decorator(subcls):
            cls._registry[name] = subcls        # 写表：  "act" → ACTConfig
            return subcls                        # 原样返回类，不破坏类本身
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        """根据名字实例化对应的子类"""
        subcls = cls._registry[name]            # 查表：  "act" → ACTConfig
        return subcls(**kwargs)

    @property
    def type(self) -> str:
        """反向查表：ACTConfig → "act" """
        for name, subcls in self._registry.items():
            if subcls is self.__class__:
                return name
        raise ValueError("未注册的类")


# ── 第二步：注册两个子类 ─────────────────────────────────────────────────────

@Registry.register("act")              # 等价于：Registry._registry["act"] = ACTConfig
@dataclass
class ACTConfig(Registry):
    chunk_size: int = 100
    use_vae: bool = True


@Registry.register("diffusion")        # 等价于：Registry._registry["diffusion"] = DiffusionConfig
@dataclass
class DiffusionConfig(Registry):
    num_steps: int = 50
    beta_schedule: str = "cosine"


# ── 第三步：演示效果 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 查看注册表内容
    print("注册表：", Registry._registry)
    # 注册表： {'act': <class 'ACTConfig'>, 'diffusion': <class 'DiffusionConfig'>}

    # 2. 用字符串名字动态创建实例（比如从配置文件读到 "type": "act"）
    policy_name = "act"                         # 假设从 config.json 读到的
    cfg = Registry.create(policy_name, chunk_size=50)
    print(cfg)
    # ACTConfig(chunk_size=50, use_vae=True)

    # 3. 反向：从实例拿回注册名（序列化时用）
    print(cfg.type)
    # act

    # 4. 新增策略时，框架代码完全不用改，只需在新文件里加装饰器
    @Registry.register("tdmpc")
    @dataclass
    class TDMPCConfig(Registry):
        horizon: int = 5

    print("注册表：", Registry._registry)
    # 注册表： {'act': ..., 'diffusion': ..., 'tdmpc': ...}
