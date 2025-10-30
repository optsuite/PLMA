import os
import torch
from natsort import natsorted
import pandas as pd
import glob

def load_qaplib(directory, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), warmup=True):
    """
    A generator function to load all problem instances under a QAPLIB directory.
    warmup: if True, the first instance is yielded once for warmup.
    """
    problem_files = natsorted(glob.glob(os.path.join(directory, '*.dat')))
    print(f"Found {len(problem_files)} problem instances in directory '{directory}'.")

    for idx, file_path in enumerate(problem_files):
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


def load_tai(data_root, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), warmup=True):
    """
    A generator function to load the 'tai' dataset.
    warmup: if True, the first instance is yielded once for warmup.
    """
    opt_path = os.path.join(data_root, 'Tai_LB.csv')
    df_opt = pd.read_csv(opt_path)
    df_opt.set_index('problem_name', inplace=True)

    pattern = os.path.join(data_root, '**', 'tai*.dat')
    problem_files = natsorted(glob.glob(pattern, recursive=True))
    first = True
    for filepath in problem_files:
        file_name = os.path.basename(filepath)
        problem_name = os.path.splitext(file_name)[0]
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