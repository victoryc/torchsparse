import torch
import torchsparse_engine


def convert_neighbor_map(neighbor_map):
    batch_indices, point_indices = torch.where(neighbor_map != -1)
    if neighbor_map.device.type == 'cuda':
        map_converted = torchsparse_engine.convert_map_forward(
            neighbor_map.int(), batch_indices.int(), point_indices.int())
    elif neighbor_map.device.type == 'cpu':
        map_converted = torchsparse_engine.cpu_convert_map_forward(
            neighbor_map.int(), batch_indices.int(), point_indices.int())
    else:
        device = neighbor_map.device
        map_converted = torchsparse_engine.cpu_convert_map_forward(
            neighbor_map.int().cpu(),
            batch_indices.int().cpu(),
            point_indices.int().cpu())
        map_converted = map_converted.to(device)
    nmap_offset = torch.sum(neighbor_map != -1, 1)
    return map_converted.int().contiguous(), nmap_offset.int().contiguous()
