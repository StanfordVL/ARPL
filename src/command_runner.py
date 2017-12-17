import os
import subprocess
import argparse
import queue
from multiprocessing.pool import ThreadPool
import argparse

def generate_commands():
    all_commands = []
    envs = range(4)
    runs = range(1,3)
    lows = [0, 5, 10, 15]
    highs = [5, 10, 15, 20]
    preamble = 'source activate arpl; cd ~/ai/ARPL;'
    for env in envs:
        for run in runs:
            for index in range(len(lows)):
                experiment_name = 'env{}_{}'.format(env, run)
                com = ' python src/eval_ddpg.py --num_workers 24 {} {} {} {}'.format(experiment_name, env, lows[index], highs[index])
                all_commands.append(preamble + com)
    return all_commands

def run_command(command, log_path):
    host = host_queue.get()
    log_path = os.path.join(log_path, command.split(';')[-1].replace(' ', '_').replace('/','_') + '.log')
    with open(log_path, 'w') as f:
        output = subprocess.run(['ssh', host,  "{}".format(command)], stdout=f, stderr=f)
    host_queue.put(host)
    ret_code = output.returncode
    if ret_code == 0:
        return '[Success] {}'.format(command)
    else:
        return '[Failed] {}'.format(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run commands remotely')
    parser.add_argument('--log_path', metavar='log_path', type=str, default='logs',
                        help='path of all log files')

    args = parser.parse_args()
    os.makedirs(args.log_path, exist_ok=True)

    hosts = ['napoli{}'.format(x + 1) for x in range(16) if x != 8]
    host_queue = queue.Queue()
    for host in hosts:
        host_queue.put(host)
    commands = generate_commands()
    pool = ThreadPool(processes=len(hosts))

    res_coll = []
    for command in commands:
        res = pool.apply_async(run_command, [command, args.log_path])
        res_coll.append(res)

    for res in res_coll:
        print(res.get())



