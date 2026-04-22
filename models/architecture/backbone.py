# 自定义模型解析器 (backbone.py)
# 重写 Ultralytics 的 YAML 解析逻辑，支持加载 CoordAtt 和 HyperACE 等自定义算子。

import math
import torch
import torch.nn as nn
from copy import deepcopy

# 导入HyperACE
from .hyperace_ops import HyperACE_Module
# 导入Coordinate Attention
from .ca_ops import CoordAtt 
# 导入 Ultralytics 模块
try:
    from ultralytics.nn.modules import (Conv, C2f, SPPF, Concat, Bottleneck, 
                                        Detect, Segment, Pose, Classify, OBB)
except ImportError:
    from ..common import Conv, C2f, SPPF, Concat, Bottleneck, Detect
    try:
        from ultralytics.nn.modules import OBB
    except:
        pass

def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch, verbose=True):
    import ast
    
    # 修复整数 ch 索引报错
    if isinstance(ch, int):
        ch = [ch]

    try:
        nc = int(d['nc'])
    except:
        nc = 1 
    gd, gw = d['depth_multiple'], d['width_multiple']
    
    layers, save, c2 = [], [], ch[-1]
    
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  
        if m in (Conv, SPPF, C2f, Bottleneck, CoordAtt):
            c1, c2 = ch[f], args[0]
            if c2 != float('inf'):
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m is C2f:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is HyperACE_Module:
            c1, c2 = ch[f], args[0]
            if c2 != float('inf'):
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        
        # OBB 参数准备: [nc, ne, ch]
        elif m is OBB:
            args.append([ch[x] for x in f])

        # Detect/Segment 参数准备
        elif m in (Detect, Segment, Pose, Classify):
            args.append([ch[x] for x in f]) 
            if isinstance(args[1], int): 
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        try:
            if n > 1:
                m_ = nn.Sequential(*(m(*args) for _ in range(n)))
            else:
                if m is Detect:
                    m_ = m(nc=args[0], ch=args[-1])
                
                elif m is OBB:
                    m_ = m(nc=args[0], ne=args[1], ch=args[2])
                    
                else:
                    m_ = m(*args)
        except Exception as e:
            print(f"\n❌ Layer {i} ({m.__name__}) Instantiation Failed!")
            print(f"   Args: {args}")
            raise e

        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        
        if verbose:
            print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
        
    return nn.Sequential(*layers), sorted(save)

class DetectionModel(nn.Module):
    def __init__(self, cfg='yolov8s_hyperace_ca.yaml', ch=3, nc=None, verbose=True): 
        super().__init__()
        import yaml
        
        with open(cfg, encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)

        self.yaml['ch'] = ch
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)
        
        # 初始化 Stride
        m = self.model[-1]
        if isinstance(m, (Detect, OBB)):
            s = 256
            m.nc = nc if nc else self.yaml['nc']
            
            try:
                out = self.forward(torch.zeros(1, ch, s, s))
                if isinstance(out, dict): out = out['one']
                if isinstance(out, (list, tuple)): 
                    m.stride = torch.tensor([s / x.shape[-2] for x in out])
            except:
                m.stride = torch.tensor([8.0, 16.0, 32.0])
            
            if hasattr(m, 'bias_init'):
                m.bias_init()

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x