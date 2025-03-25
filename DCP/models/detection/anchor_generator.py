import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import math

class AnchorGenerator(nn.Module):
    """锚框生成器"""
    def __init__(self, 
                 sizes: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),),
                 aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
                 strides: Tuple[int, ...] = (16,),
                 offset: float = 0.5):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.offset = offset
        
        # 验证输入参数
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = tuple((a,) for a in aspect_ratios)
            
        self.num_anchors_per_location = [len(s) * len(a) for s, a in zip(sizes, aspect_ratios)]
        
    def forward(self, image_list: List[torch.Tensor], feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        生成锚框
        Args:
            image_list: 输入图像列表
            feature_maps: 特征图列表
        Returns:
            anchors: 锚框列表
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list[0].shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        
        strides = [[int(image_size[0] / g[0]), int(image_size[1] / g[1])] for g in grid_sizes]
        
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides, dtype, device)
        
        anchors: List[torch.Tensor] = []
        for i, (image_height, image_width) in enumerate(image_list.shape[-2:] for image_list in image_list):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
            
        return anchors
        
    def grid_anchors(self, grid_sizes: List[Tuple[int, int]], strides: List[List[int]],
                     dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
        """生成网格锚框"""
        anchors = []
        for size, aspect_ratio, stride, grid_size in zip(
            self.sizes, self.aspect_ratios, strides, grid_sizes
        ):
            grid_height, grid_width = grid_size
            stride_height, stride_width = stride
            
            # 生成网格点
            shifts_x = torch.arange(
                self.offset * stride_width,
                grid_width * stride_width,
                step=stride_width,
                dtype=dtype,
                device=device,
            )
            shifts_y = torch.arange(
                self.offset * stride_height,
                grid_height * stride_height,
                step=stride_height,
                dtype=dtype,
                device=device,
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
            # 生成基础锚框
            anchors_per_grid = []
            for size_per_grid, aspect_ratio_per_grid in zip(size, aspect_ratio):
                h = size_per_grid / math.sqrt(aspect_ratio_per_grid)
                w = size_per_grid * math.sqrt(aspect_ratio_per_grid)
                anchors_per_grid.append(
                    torch.tensor(
                        [-w / 2, -h / 2, w / 2, h / 2],
                        dtype=dtype,
                        device=device,
                    )
                )
            anchors_per_grid = torch.stack(anchors_per_grid, dim=0)
            
            # 将基础锚框应用到所有网格点
            anchors_per_grid = anchors_per_grid.view(1, -1, 4)
            shifts = shifts.view(-1, 1, 4)
            anchors_per_grid = anchors_per_grid + shifts
            anchors_per_grid = anchors_per_grid.view(-1, 4)
            
            anchors.append(anchors_per_grid)
            
        return anchors
        
    def num_anchors_per_location(self) -> List[int]:
        """获取每个位置的锚框数量"""
        return self.num_anchors_per_location

class AnchorGeneratorWithAspectRatio(AnchorGenerator):
    """支持自定义宽高比的锚框生成器"""
    def __init__(self, 
                 sizes: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),),
                 aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
                 strides: Tuple[int, ...] = (16,),
                 offset: float = 0.5,
                 custom_aspect_ratios: Dict[str, Tuple[float, ...]] = None):
        super().__init__(sizes, aspect_ratios, strides, offset)
        self.custom_aspect_ratios = custom_aspect_ratios or {}
        
    def grid_anchors(self, grid_sizes: List[Tuple[int, int]], strides: List[List[int]],
                     dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
        """生成网格锚框，支持自定义宽高比"""
        anchors = []
        for size, aspect_ratio, stride, grid_size in zip(
            self.sizes, self.aspect_ratios, strides, grid_sizes
        ):
            grid_height, grid_width = grid_size
            stride_height, stride_width = stride
            
            # 生成网格点
            shifts_x = torch.arange(
                self.offset * stride_width,
                grid_width * stride_width,
                step=stride_width,
                dtype=dtype,
                device=device,
            )
            shifts_y = torch.arange(
                self.offset * stride_height,
                grid_height * stride_height,
                step=stride_height,
                dtype=dtype,
                device=device,
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
            # 生成基础锚框
            anchors_per_grid = []
            for size_per_grid, aspect_ratio_per_grid in zip(size, aspect_ratio):
                # 检查是否有自定义宽高比
                if str(size_per_grid) in self.custom_aspect_ratios:
                    custom_ratios = self.custom_aspect_ratios[str(size_per_grid)]
                    for ratio in custom_ratios:
                        h = size_per_grid / math.sqrt(ratio)
                        w = size_per_grid * math.sqrt(ratio)
                        anchors_per_grid.append(
                            torch.tensor(
                                [-w / 2, -h / 2, w / 2, h / 2],
                                dtype=dtype,
                                device=device,
                            )
                        )
                else:
                    h = size_per_grid / math.sqrt(aspect_ratio_per_grid)
                    w = size_per_grid * math.sqrt(aspect_ratio_per_grid)
                    anchors_per_grid.append(
                        torch.tensor(
                            [-w / 2, -h / 2, w / 2, h / 2],
                            dtype=dtype,
                            device=device,
                        )
                    )
            anchors_per_grid = torch.stack(anchors_per_grid, dim=0)
            
            # 将基础锚框应用到所有网格点
            anchors_per_grid = anchors_per_grid.view(1, -1, 4)
            shifts = shifts.view(-1, 1, 4)
            anchors_per_grid = anchors_per_grid + shifts
            anchors_per_grid = anchors_per_grid.view(-1, 4)
            
            anchors.append(anchors_per_grid)
            
        return anchors 