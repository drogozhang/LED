import torch
import accelerate
import torch.distributed as dist
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

if __name__ == '__main__':
    accelerator = accelerate.Accelerator()

    eval_dataset = TensorDataset(torch.arange(1, 20), )
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)

    dist_dataloader = accelerator.prepare(eval_dataloader)

    all_gathered = []
    for batch in dist_dataloader:
        print(dist.get_rank(), batch[0])

        gathered = accelerator.gather(batch[0])
        if accelerator.is_local_main_process:
            all_gathered.append(gathered)
    if accelerator.is_local_main_process:
        print(torch.cat(all_gathered, dim=-1))
        print(len(eval_dataset))
        print(torch.cat(all_gathered, dim=-1)[:len(eval_dataset)])








