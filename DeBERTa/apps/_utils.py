import torch
from collections import OrderedDict, Mapping, Sequence

import smdistributed.modelparallel
import smdistributed.modelparallel.torch as smp

def merge_distributed(data_list, max_len=None):
  # if torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
  #   world_size = torch.distributed.get_world_size()
  # else:
  #   world_size = 1
  world_size = smp.dp_size()
  merged = []
  def gather(data):
    data_size = [torch.zeros(data.dim(), dtype=torch.int).to(data.device) for _ in range(world_size)]
    torch.distributed.all_gather(data_size, torch.tensor(data.size()).to(data_size[0]))
    data_chunks = [torch.zeros(tuple(s.cpu().numpy())).to(data) for s in data_size]
    data_chunks[data.device.index] = data
    for i,_chunk in enumerate(data_chunks):
      torch.distributed.broadcast(_chunk, src=i)
    return data_chunks

  if smp.dp_rank() == 0:
    print('data_list: ', data_list)
  for data in data_list:
    # if torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
    if smp.dp_size() > 1:
      if isinstance(data, Sequence):
        data_chunks = []
        for d in data:
          chunks_ = gather(d)
          data_ = torch.cat(chunks_)
          data_chunks.append(data_)
        merged.append(data_chunks)
      else:
        if smp.dp_rank() == 0:
          print('Processing data')
        _chunks = gather(data)
        if smp.dp_rank() == 0:
          print('Running gather(data)')
        merged.extend(_chunks)
        if smp.dp_rank() == 0:
          print('extend data')
    else:
      merged.append(data)
  if smp.dp_rank() == 0:
    print('return data')
  return join_chunks(merged, max_len)

def join_chunks(chunks, max_len=None):
  if not isinstance(chunks[0], Sequence):
    merged = torch.cat([m.cpu() for m in chunks])
    if max_len is not None:
      return merged[:max_len]
    else:
      return merged
  else:
    data_list=[]
    for d in zip(*chunks):
      data = torch.cat([x.cpu() for x in d])
      if max_len is not None:
        data = data[:max_len]
      data_list.append(data)
    return data_list

