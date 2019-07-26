import subprocess
import time

import ray


def dispatch_command_strings(commands, cpus_per_command=1, gpus_per_command=0, pause_time=0):
    if not ray.is_initialized():
        raise Exception('Ray must be initialised to dispatch commands')

    while len(commands) > 0:
        resources = ray.available_resources()
        if 'CPU' in resources and resources['CPU'] >= cpus_per_command:
            if gpus_per_command == 0 or ('GPU' in resources and resources['GPU'] >= gpus_per_command):
                command = commands.pop() + f' -r localhost:9003 &'
                print('Starting', command)
                subprocess.call(command, shell=True)
                time.sleep(pause_time)


if __name__ == '__main__':
    ray.init(redis_address='localhost:9003')
    print('Nodes', ray.nodes())
    print('Tasks', ray.tasks())
    print('Total Resources', ray.cluster_resources())
    print('Avalible Resources', ray.available_resources())

    commands = [f'python example_ray_process.py -n {i}' for i in range(10)]

    dispatch_command_strings(commands, cpus_per_command=1, pause_time=1.)
