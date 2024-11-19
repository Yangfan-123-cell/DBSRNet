from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--test_structure_path', type=str, default='/home/dell/桌面/coding/2022/LY/qads3/super-resolved_images/', help='path to reference images')
        self._parser.add_argument('--test_lbp_path', type=str, default='/home/dell/桌面/coding/2022/LY/qads3/super-resolved_images/', help='path to distortion images')
        self._parser.add_argument('--test_list', type=str, default='/home/dell/桌面/coding/2022/LY/qads3/test-score4.txt', help='training data')
        
        self._parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self._parser.add_argument('--test_file_name', type=str, default='score-qads-single-cnn6.txt', help='txt path to save results')
        self._parser.add_argument('--n_ensemble', type=int, default=20, help='crop method for test: five points crop or nine points crop or random crop for several times')
        self._parser.add_argument('--flip', type=bool, default=False, help='if flip images when testing')
        self._parser.add_argument('--resize', type=bool, default=False, help='if resize images when testing')
        self._parser.add_argument('--size', type=int, default=224, help='the resize shape')
        self.is_train = False
