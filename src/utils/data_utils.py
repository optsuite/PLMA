import os
import torch
from natsort import natsorted
import pandas as pd
import glob

def load_qaplib(directory, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), warmup=True):
    """
    一个生成器函数 用于加载QAPLIB目录下的所有问题实例。
    warmup: 如果为True，第一个实例会先yield一次用于预热
    """
    problem_files = sorted(glob.glob(os.path.join(directory, '*.dat')))
    print(f"在目录 '{directory}' 中发现了 {len(problem_files)} 个问题实例。")

    for idx, file_path in enumerate(problem_files):
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip().split()
            n = int(content[0])
            opt = int(content[1])
            data = list(map(float, content[2:]))
            flows = torch.tensor(data[: n * n]).reshape(n, n).to(device)
            distances = torch.tensor(data[n * n :]).reshape(n, n).to(device)
            problem_name = os.path.basename(file_path)

            if idx == 0 and warmup:
                yield f"warmup_{problem_name}", (n, opt, flows, distances)
            
            yield problem_name, (n, opt, flows, distances)

        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            continue


def load_tai(data_root, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), warmup=True):
    """
    一个生成器函数，用于加载'tai'数据集。
    warmup: 如果为True，第一个实例会先yield一次用于预热
    """
    opt_path = os.path.join(data_root, 'Tai_LB.csv')
    df_opt = pd.read_csv(opt_path)
    df_opt.set_index('problem_name', inplace=True)
    
    first = True
    for subdir, _, files in os.walk(data_root):
        for file_name in natsorted(files):
            if file_name == 'Tai_LB.csv':
                continue

            problem_name = os.path.splitext(file_name)[0]
            filepath = os.path.join(subdir, file_name)
            opt = df_opt.loc[problem_name, 'optimal_value']
            with open(filepath, 'r') as f:
                content = f.read().strip().split()
                n = int(content[0])
                data = list(map(float, content[1:]))
                A = torch.tensor(data[: n * n]).reshape(n, n).to(device)
                B = torch.tensor(data[n * n :]).reshape(n, n).to(device)

            if first and warmup:
                yield f"warmup_{problem_name}", (n, opt, A, B)
                first = False
            
            yield problem_name, (n, opt, A, B)