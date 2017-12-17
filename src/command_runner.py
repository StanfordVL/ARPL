import subprocess
import argparse
import thread
import queue
from multiprocessing.pool import ThreadPool

def generate_commands():
    return ['source activate arpl; ls -l .' for x in range(10)]

def run_command(command):
    host = host_queue.pop()
    output = subprocess.run('ssh {} "{}"'.format(host, command))
    host_queue.push(host)
    ret_code = output.returncode
    if ret_code == 0:
        return '[Success] {}'.format(command)
    else:
        return '[Failed] {}'.format(command)

if __name__ == '__main__':
    hosts = ['napoli{}'.format(x + 1) for x in range(16)]
    host_queue = queue.Queue()
    for host in hosts:
        host_queue.append(host)
    commands = generate_commands()
    pool = ThreadPool(processes=len(hosts))

    res_coll = []
    for command in commands:
        res = p.apply_async(run_command, command)
        res_coll.append(res)

    for res in res_coll:
        print(res.get())



