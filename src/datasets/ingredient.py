from src.datasets.loader import DatasetFolder
from src.datasets.transform import with_augment, without_augment
from torch.utils.data import DataLoader

def get_dataloader(sets, args, sampler=None, shuffle=True, pin_memory=False):
    if sampler is not None:
        loader = DataLoader(sets, batch_sampler=sampler,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        loader = DataLoader(sets, batch_size=args.batch_size_loader, shuffle=shuffle,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    return loader

def get_dataset(split, args, aug=False, out_name=False):
    if aug:
        transform = with_augment(84, disable_random_resize=args.disable_random_resize,
                                 jitter=args.jitter)
    else:
        transform = without_augment(84, enlarge=args.enlarge)
    print("path", args.root + '/' + args.dataset_path)
    sets = DatasetFolder(args.root + '/' + args.dataset_path, args.split_dir, split, transform, out_name=out_name)
    return sets
