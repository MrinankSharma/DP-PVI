import argparse

import ray

argparser = argparse.ArgumentParser()
argparser.add_argument('-r', '--ray', required=True)
argparser.add_argument('-n', '--number', required=True)
args = argparser.parse_args()

ray.init(redis_address=args.ray)


@ray.remote(num_cpus=1)
def test_process(number):
    print('Start', number)
    a = 0
    for i in range(2_000_000_00):
        a = a + 1
    print('End', number)


print(ray.get(test_process.remote(args.number)))
