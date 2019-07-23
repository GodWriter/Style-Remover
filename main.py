import os

from dataloder import Dataset
from solver import Solver
from config import parse_args, read_conf_file

args_cmd = parse_args()
args_yml = read_conf_file(args_cmd.config)

data = Dataset(args_yml)

if __name__ == '__main__':
    module = args_cmd.module

    os.environ['CUDA_VISIBLE_DEVICES'] = args_cmd.GPU

    if module == 'test_dataset':
        data.test_dataset()
    elif module == 'create_dataset':
        data.create_dataset()
    elif module == 'train':
        solver = Solver(args_yml)
        solver.train()
    elif module == 'test_dir':
        solver = Solver(args_cmd)
        solver.test_dir()

