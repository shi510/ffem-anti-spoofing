import argparse
import os

import celeba_spoof


parser = argparse.ArgumentParser()
sub_parser = parser.add_subparsers(dest='cmd')
celeba_spoof_parser = sub_parser.add_parser('celeba_spoof')
celeba_spoof_parser.add_argument('--dir')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.cmd == 'celeba_spoof':
        train_path = os.path.join(args.dir, 'Data', 'train')
        test_path = os.path.join(args.dir, 'Data', 'test')
        if not os.path.exists(train_path):
            print('train path not exists: {}'.format(train_path))
        if not os.path.exists(test_path):
            print('test path not exists: {}'.format(test_path))
        if os.path.isdir(args.dir):
            celeba_spoof.save_bbox_to_json(
                args.dir,
                os.path.join('Data', 'train'),
                'celeba_spoof_train.json')
            celeba_spoof.save_bbox_to_json(
                args.dir,
                os.path.join('Data', 'test'),
                'celeba_spoof_test.json')
        else:
            print('{} not exists.'.format(args.dir))
    else:
        print('{} is not support.'.format(args.cmd))
