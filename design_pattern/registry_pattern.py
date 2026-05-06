

from dataclasses import dataclass
from typing import ClassVar, Dict


class Registry:
    _registry: ClassVar[Dict[str, type]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(subcls):
            cls._registry[name] = subcls 
            return subcls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs):
        """根据名字实例化对应的子类"""
        subcls = cls._registry[name]
        return subcls(**kwargs)
    
    @property
    def type(self) -> str:
        """获取当前类在注册表中的名字"""
        for name, subcls in self._registry.items():
            if subcls == type(self):
                return name
            raise ValueError(f"{type(self)} not registered")
    


@Registry.register("act")
@dataclass
class ACTConfig(Registry):
    chunk_size: int = 50
    use_vae: bool = True


@Registry.register("diffusion")
@dataclass
class DiffusionConfig(Registry):
    num_steps: int = 50
    beta_schedule: str = "cosine"




if __name__ == "__main__":
    #1--查看注册表内容
    print("注册表：", Registry._registry)

    #2--用字符串名字动态创建实例
    policy_name = "act"                        
    cfg = Registry.create(policy_name, chunk_size=50)
    print(cfg)

    #3--反向：从实例拿回注册名
    print(cfg.type)

    #3--新增策略
    @Registry.register("tdmpc")
    @dataclass
    class TDMPCConfig(Registry):
        horizon: int = 5
    print("注册表：", Registry._registry)