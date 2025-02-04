import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import is_image_file
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Optional, Callable, List, Tuple, Dict, Any


class MVTecAD(VisionDataset):
    """
    Args:
        root (string): Root directory of the MVTec AD Dataset
        category (string, optional): One of the MVTec AD Dataset names
        train (bool, optional): If true, use the train dataset, otherwise the test dataset.
        transform (callable, optional): A function/transform that takes in an PIL image
                                        and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        mask_transform (callable, optional): A function/transform that takes in the mask and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        pin_memory (bool, optional): If true, load all images into memory in this class. Otherwise, only image paths are kept.
    Attributes:
        subset_name (str): name of the loaded subset.
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        image_paths (list): List of image paths.
        mask_paths (list): List of mask paths.
        data (list): List of PIL images. not named with 'images' for consistence with common dataset, such as cifar.
        masks (list): List of PIL masks. mask is of the same size of image and indicate the anomaly pixels.
        targets (list): The class_index value for each image in the dataset.
    Note:
        The normal class index is 0.
        The abnormal class indexes are assigned 1 or higher alphabetically.
    """

    normal_str = 'good'
    mask_str = 'ground_truth'
    train_str = 'train'
    test_str = 'test'
    compress_ext = '.tar.xz'
    image_size = (900, 900)

    def __init__(self,
                root: str,
                category: str,
                train: bool=True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                mask_transform: Optional[Callable] = None,
                pin_memory=False):

        super(MVTecAD, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.mask_transform = mask_transform
        self.pin_memory = pin_memory

        self.dataset_root = os.path.join(self.root)
        self.category = category.lower()
        self.subset_root = os.path.join(self.dataset_root, self.category)
        self.subset_split = os.path.join(self.subset_root, self.train_str if self.train else self.test_str)

        if not os.path.exists(self.subset_root):
            raise FileNotFoundError('subset {} is not found, please set download=True to download it.')

        # get image classes and corresponding targets
        self.classes, self.class_to_idx = self._find_classes(self.subset_split)

        # get image paths, mask paths and targets
        self.image_paths, self.mask_paths, self.targets = self._find_paths(self.subset_split, self.class_to_idx)
        if self.__len__() == 0:
            raise FileNotFoundError("found 0 files in {}\n".format(self.subset_split))

        # pin memory (usually used for small datasets)
        if self.pin_memory:
            self.data = self._load_images('RGB', self.image_paths)
            self.masks = self._load_images('L', self.mask_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        '''
        get item iter.
        :param idx (int): idx
        :return: (tuple): (image, mask, target) where target is index of the target class.
        '''
        # get image, mask and target of idx
        if self.pin_memory:
            image, mask = self.data[idx], self.masks[idx]
        else:
            image, mask = self._pil_loader('RGB', self.image_paths[idx]), self._pil_loader('L', self.mask_paths[idx])
        target = self.targets[idx]

        # apply transform
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.transform(mask)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, mask, target

    def __len__(self) -> int:
        """
        return the length of the dataset
        """
        return len(self.targets)

    def extra_repr(self):
        split = self.train_str if self.train else self.test_str
        return 'using data: {data}\nsplit: {split}'.format(data=self.category, split=split)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            classes: (unknown) of the class names sorted alphabetically.    #FIXME: what is (unknown)?
            class_to_idx: (dict): Dict with items (class_name, class_index).#FIXME: Figure out

        Ensures:
            No class is a subdirectory of another.
        """
        # classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes = []
        for d in os.scandir(path=dir):
            if d.is_dir():
                classes.append(d.name)
        classes.remove(self.normal_str)
        classes = [self.normal_str] + classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _find_paths(self,
                    folder: str,
                    class_to_idx: Dict[str, int]) -> Tuple[Any, Any, Any]:
        '''
        find image paths, mask paths and corresponding targets
        :param folder: folder/class_0/*.*
                       folder/class_1/*.*
        :param class_to_idx: dict of class name and corresponding label
        :return: image paths, mask paths, targets
        '''
        # define variables to fill
        image_paths, mask_paths, targets = [], [], []

        # define path find helper
        def find_mask_from_image(target_class, image_path):
            '''
            find mask path according to image path
            :param target_class: target class
            :param image_path: image path
            :return: None or mask path
            '''
            if target_class is self.normal_str:
                mask_path = None
            else:
                # only test data have mask images
                mask_path = image_path.replace(self.test_str, self.mask_str)
                fext = '.' + fname.split('.')[-1]
                mask_path = mask_path.replace(fext, '_mask' + fext)
            return mask_path

        # find
        for target_class in class_to_idx.keys():
            class_idx = class_to_idx[target_class]
            target_folder = os.path.join(folder, target_class)
            for root, _, fnames in sorted(os.walk(target_folder, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        # get image
                        image_paths.append(os.path.join(root, fname))
                        # get mask
                        mask_paths.append(find_mask_from_image(target_class, image_paths[-1]))
                        # get target
                        targets.append(class_idx)

        return image_paths, mask_paths, targets

    def _pil_loader(self, mode: str, path: str):
        '''
        load PIL image according to path.
        :param mode: PIL option, 'RGB' or 'L'
        :param path: image path, None refers to create a new image
        :return: PIL image
        '''
        if path is None:
            image = Image.new(mode, size=self.image_size)
        else:
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            # !!! directly using Image.open(mode, path) will lead to Dataloader Error inside loop !!!
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert(mode)
        return image

    def _load_images(self, mode: str, paths: List[str]) -> List[Any]:
        '''
        load images according to paths.
        :param mode: PIL option, 'RGB' or 'L'
        :param paths: paths of images to load
        :return: list of images
        '''
        images = []
        for path in paths:
            images.append(self._pil_loader(mode, path))
        return images

if __name__ == '__main__':
    def _convert_label(x):
        return 0 if x == 0 else 1
    
    # define transforms
    transform = transforms.Compose(transforms=[ transforms.Resize(size=(300, 300)),
                                                transforms.ToTensor()   ])
    target_transform = transforms.Lambda(_convert_label)
    
    # load data
    mvtec = MVTecAD(root='/home/msis/Work/dataset/MVTec_AD',        category='bottle',
                    train=True,     transform=transform,
                    target_transform=target_transform,      mask_transform=transform)
    
    # feed to data loader
    data_loader = DataLoader(dataset=mvtec,     batch_size=2,   shuffle=True,
                            num_workers=8,     pin_memory=True,    drop_last=True)
    
    for idx, (image, mask, target) in enumerate(data_loader):
        print(idx, target)