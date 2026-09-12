from pathlib import Path
import shutil

from .retrain_rgb_baseline import replace_once


def patch_valid_region(vendor):
    vendor = Path(vendor)
    dataset = vendor / 'dataset.py'
    trainer = vendor / 'ace_trainer.py'
    data = dataset.read_text()
    training = trainer.read_text()
    data = replace_once(data, 'from ace_network import Regressor',
        'from ace_network import Regressor\nfrom valid_region import validate_mask, resize_valid_mask')
    data = replace_once(data, '        root_dir = Path(root_dir)',
        "        root_dir = Path(root_dir)\n"
        "        mask_path = root_dir / 'valid_mask.npy'\n"
        "        self.valid_region = validate_mask(np.load(mask_path)) if mask_path.exists() else None")
    data = replace_once(data, '        image = self._load_image(idx)',
        '        image = self._load_image(idx)\n'
        '        if self.valid_region is not None and self.valid_region.shape != image.shape[:2]:\n'
        "            raise ValueError('Valid mask and stored RGB dimensions differ')")
    data = replace_once(data, '        image_mask = torch.ones((1, image.size[1], image.size[0]))',
        '        image_mask = torch.ones((1, image.size[1], image.size[0]))\n'
        '        if self.valid_region is not None:\n'
        '            image_mask = torch.from_numpy(resize_valid_mask(self.valid_region, image.size[1], image.size[0]))[None]')
    data = replace_once(data, '        image_mask = image_mask > 0', '        image_mask = image_mask >= 1 - 1e-6')
    training = replace_once(training, 'import os\n', 'import os\nfrom valid_region import grid_valid_region\n')
    training = replace_once(training,
        '                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)\n'
        '                    image_mask_B1HW = image_mask_B1HW.bool()',
        '                    image_mask_B1HW = grid_valid_region(image_mask_B1HW, H, W)')
    training = replace_once(training,
        '                    if image_mask_B1HW.sum() == 0:\n                        continue',
        '                    if image_mask_B1HW.sum() == 0:\n'
        "                        raise ValueError('Image has no valid GLACE output pixel centers')")
    shutil.copyfile(Path(__file__).with_name('valid_region.py'), vendor / 'valid_region.py')
    dataset.write_text(data)
    trainer.write_text(training)
