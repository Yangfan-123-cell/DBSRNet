from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--n_epoch', type=int, default=201, help='total epoch for training')
        self._parser.add_argument('--save_interval', type=int, default=2, help='interval for saving models')
        self._parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
        self._parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self._parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self._parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
        self._parser.add_argument('--T_max', type=int, default=50, help="cosine learning rate period (iteration)")
        self._parser.add_argument('--eta_min', type=int, default=0, help="mininum learning rate")
        self._parser.add_argument('--n_ensemble', type=int, default=20, help='crop method for test: five points crop or nine points crop or random crop for several times')
        #xinjiade
        #self._parser.add_argument('--val_file_name', type=str, default='val_score_cviu.txt', help='txt path to save results')
        
        self.is_train = True