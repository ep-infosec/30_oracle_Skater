"""Utility for parsing command line arguments for testing"""

import argparse


def arg_parse(args):
    parser = create_parser()
    return parser.parse_args(args)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n', default=1000, type=int)
    parser.add_argument('--dim', default=3, type=int)
    parser.add_argument('--r_deploy_test', action='store_true')
    parser.add_argument('--python_deploy_test', action='store_true')
    return parser
