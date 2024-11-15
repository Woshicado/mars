#!python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import time

import argcomplete
import yaml
from yaml.loader import SafeLoader

SCENE_DIR_DEFAULT = 'scenes'

def main():
    with open('config.yaml', 'r') as f:
        configs = list(yaml.load_all(f, Loader=SafeLoader))[0]
    all_strategies = configs['strategies']

    # Automatically register all scenes in the default scene directory
    all_scenes = os.listdir(SCENE_DIR_DEFAULT)
    all_scenes = [entry for entry in all_scenes
                  if os.path.isdir(os.path.join(SCENE_DIR_DEFAULT, entry)) and not entry.startswith('_')]

    # And allow for manually registered scenes, which might be necessary if not using the default scene dir
    all_scenes += configs['known-scenes']
    all_scenes = list(set(all_scenes))

    args = parse([ 'all', 'paper' ] + list(all_strategies.keys()), [ 'all' ] + sorted(all_scenes))
    compile(args)


    if 'all' in args.desc:
        args.desc = all_strategies
    elif 'paper' in args.desc:
        args.desc = [ "B-classicRR", "B-brute-force", "B-EARS", "B-MARS-bu", "B-MARS",
                      "G-classicRR", "G-EARS",   "G-MARS-grad", "G-MARS-bu", "G-MARS" ]

    if 'all' in args.scenes:
        args.scenes = all_scenes


    for scene in args.scenes:
        scene_res_dir = os.path.join(args.outdir, scene, args.subdir)
        os.makedirs(scene_res_dir, exist_ok=True)

        for strat in args.desc:
            cmd = build_command_from_strat(args, configs, scene, strat)
            pprint_command(cmd, configs['sort-key'], not args.disable_colors)
            if args.run:
                run_command(args, cmd)
        print("")

    runtime(args, args.scenes, args.desc, True)

def parse(all_strategies, all_scenes):
    parser = argparse.ArgumentParser(
                    prog='render.py',
                    description='Renders all enabled configs in sequence on all scenes',
                    epilog='')

    # Script arguments
    parser.add_argument('-r','--run', action='store_true',
                        help='Execute the commands instead of only printing them')

    parser.add_argument('-S', '--scenes', metavar='SCENE', nargs='+', default='all', choices=all_scenes,
                        help=f"Scene(s) to be rendered. Possible choices: {', '.join(all_scenes)}")

    parser.add_argument('-cfg', '--render-configs', action='store', dest='desc', nargs='+', default='paper',
                        choices=all_strategies, metavar="CONFIG",
                        help=f"Configurations by ID\'s used in \'config.yaml > strategies\'. Possible choices: {', '.join(all_strategies)}")

    parser.add_argument('-b','--budget', type=int, action='store', default=300,
                        help='Sets the rendering budget')

    parser.add_argument('-c','--compile', action='store_true',
                        help='Compile mitsuba before execution')

    parser.add_argument('-v','--verbose', action='store_true',
                        help='Print rendering output to stdout. Also disables this scripts progress bar')

    parser.add_argument('--gui', action='store_true',
                        help='Run mtsgui instead of mitsuba')

    parser.add_argument('--sMin', type=float, action='store', default=0.05,
                        help='Minimum sample allocation budget per technique')

    parser.add_argument('--sMax', type=float, action='store', default=20,
                        help='Maximum sample allocation budget per technique')

    parser.add_argument('--maxDepth', type=int, action='store', default=40)

    parser.add_argument('--rrDepth', type=int, action='store', default=5,
                        help='Depth at which RR start')

    parser.add_argument('-o','--outdir', type=pathlib.Path, action='store', default='_results',
                        help='Output path')

    parser.add_argument('--subdir', action='store', default=time.strftime("%y%m%d"),
                        help='Subdirectory to put renders of this execution\'s results. Default: today\'s date in \'yymmdd\'')

    parser.add_argument('--file-name', action='store', default='out')

    parser.add_argument('--train-fraction', action='store', type=int, default=2,
                        help='(GUIDING) Denominator defining fraction of budget used for training')

    parser.add_argument('--train-budget', action='store', type=int, default=None,
                        help='(GUIDING) Explicit train budget. Disables train-fraction if given')

    parser.add_argument('--train-iterations', action='store', type=int, default=9,
                        help='(GUIDING) Number of training iterations')

    parser.add_argument('--disable-colors', action='store_true',
                        help='Disables color output of the command')

    parser.add_argument('-s','--skip', type=int, default=0, action='store',
                        help='Skips the first `n` commands')

    parser.add_argument('-d','--delete', action='store_true',
                        help='Clear the output directory of each scene before execution')

    parser.add_argument('--store-train', action='store_true',
                        help='(EARS+Bidir. MARS) Saves the training iterations images next to the output file')

    parser.add_argument('-g', '--gdb', action='store_true',
                        help='Run the commands using gdb. Useful for debugging a command without having to copy it')

    parser.add_argument('--mis-exp', type=int, action='store', default=1,
                        help='(Bidir. MARS) Exponent of the MIS power heuristic')

    parser.add_argument('--memLimit', type=int, action='store', default=24,
                        help='(EARS+MARS) Memory limit of cache. This is multiplied by the number of techniques for MARS')

    parser.add_argument('--li', action='store_true',
                        help='(NOT THOROUGHLY TESTED) Activates light-image rendering for BDPT/MARS/effmis')

    parser.add_argument('--mitsuba-path', type=pathlib.Path, action='store', default='mitsuba',
                        help='Path to the parent directory of mitsuba\'s source. Necessary for compile flag to work')

    parser.add_argument('--scene-dir', type=pathlib.Path, action='store',
                        default=SCENE_DIR_DEFAULT)

    parser.add_argument('--scene-file', type=pathlib.Path, action='store',
                        default='scene.xml')

    parser.add_argument('--outlierFactor', type=float, action='store', default=50,
                        help='(BRUTE-FORCE) Changes outlier rejection factor. Higher = Less rejection')

    parser.add_argument('-j', action='store', default=16,
                        help='Number of threads to start for compilation and building of mitsuba')

    parser.add_argument('--partial', action='store', default=-1, type=int, metavar='X',
                        help='Store partial image every X seconds')

    argcomplete.autocomplete(parser)

    return parser.parse_args()

def ndict(d):
    return d if d is not None else {}

def delete(args, dir):
    if args.delete and args.run:
        shutil.rmtree(dir, ignore_errors=True)

def runtime(args, scenes, strategies, stdout=False):
    GREEN='\033[0;32m'
    NC='\033[0m' # No color

    runtime_s = len(strategies) * len(scenes) * args.budget
    hours = runtime_s // 3600
    remaining_seconds = runtime_s % 3600
    minutes = remaining_seconds // 60
    if stdout:
        print(f"Estimated runtime is {GREEN}{hours}h {minutes}m{NC}")
    return hours, minutes

def build_command_from_strat(args, configs, scene, strategy):
    strategies = configs['strategies']
    strat_config = strategies[strategy]
    integrator = strat_config['flags']['fixed']['integrator']

    # First initialize the provided flags with the defaults
    flags = ndict(strat_config['flags']['defaults'])

    # Overwrite with provided args
    provided = {
                    'maxDepth': args.maxDepth,
                    'rrDepth': args.rrDepth,
                    'budget': args.budget,
                    'sfExp': args.mis_exp,
                    'li': str(args.li).lower(),
                    'save': str(args.store_train).lower(),
                    'tBudget': args.budget // args.train_fraction
                                    if args.train_budget is None else args.train_budget,
                    'rBudget': args.budget - (args.budget // args.train_fraction)
                                    if args.train_budget is None else args.budget,
                    'trainIter': args.train_iterations,
                    'memLimit': args.memLimit,
                    'outlierFactor': args.outlierFactor,
                    'sMin': args.sMin,
                    'sMax': args.sMax
                }

    flags = flags | provided

    # Overwrite every fixed with the fixed values
    flags = flags | ndict(strat_config['flags']['fixed'])

    # Remove all unsupported flags
    integ_flags = set(configs['integrator-flags'][integrator])\
                     if configs['integrator-flags'][integrator] is not None else set()
    general_flags = set(configs['general-flags'])\
                     if configs['general-flags'] is not None else set()
    unsupported = strat_config['flags']['unsupported']\
                     if strat_config['flags']['unsupported'] is not None else []
    unsupported += list(flags.keys() - integ_flags - general_flags)
    for key in unsupported:
        flags.pop(key, None)

    result_dir = os.path.join(args.outdir, scene, args.subdir, strategy)
    delete(args, result_dir)

    arguments  = []
    if args.gdb:
        arguments += ['gdb', '--args']
    arguments += ['mtsgui'] if args.gui else ['mitsuba']
    if args.partial > 0:
        arguments += [ f'-r {args.partial}' ]
    arguments += [ f'-D{s}={v}' for (s,v) in flags.items()]
    arguments += [ f'{os.path.join(args.scene_dir, scene, args.scene_file)}']
    arguments += [ '-o', os.path.join(result_dir, f'{args.file_name}.exr')]

    return arguments

def custom_sort_key(string, weights):
    # I might want to use a dict in a later iteration instead of enumerating a list, who knows
    weight = float('inf')

    for w, word in enumerate(weights):
        if f"{word}=" in string:
            weight = min(weight, w)
            return weight # Remove if more than one possible

    return weight

def pprint_command(command, sort_key, colorize=True):
    # COLOR DEFINITIONS
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    BGREEN='\033[0;92m'
    BRED='\033[0;33m'
    NC='\033[0m' # No color

    # Sort command arguments for visual pleasure
    keep_first, keep_last = 1, 3
    first = command[:keep_first]
    middle = command[keep_first:-keep_last]
    last = command[-keep_last:]

    middle.sort(key=lambda x: custom_sort_key(x, sort_key))

    command = first + middle + last

    if colorize:
        colorize_tf = lambda s: s.replace('true', f'{BGREEN}true {NC}').replace('false', f'{RED}false{NC}')
        colorize_numbers = lambda s: re.sub(r'(?<!L)(\d+)', f"{GREEN}"r'\1'f"{NC}", s)
        colorize_rrsStrat = lambda s: re.sub(r'rrsStrategy=(\w+)', f"rrsStrategy={YELLOW}"r'\1'f"{NC}", s)
        colorize_integrators = lambda s: re.sub(r'integrator=(\w+)', f"integrator={YELLOW}"r'\1'f"{NC}", s)
        colorize_splitConf = lambda s: re.sub(r'splitConf=(\w+)', f"splitConf={YELLOW}"r'\1'f"{NC}", s)
        colorize_keys = lambda s: re.sub(r'-D(\w+)=', f"-D{BLUE}"r'\1'f"{NC}=", s)
        colorize_scene = lambda s: re.sub(r'/(.+)/scene\.xml', f"/{BRED}"r'\1'f"{NC}/scene.xml", s)
        command = list(map(colorize_numbers, command))
        command = list(map(colorize_tf, command))
        command = list(map(colorize_rrsStrat, command))
        command = list(map(colorize_integrators, command))
        command = list(map(colorize_splitConf, command))
        command = list(map(colorize_scene, command))
        command = list(map(colorize_keys, command))

    print(' '.join(x for x in command))

def run_command(args, command):
    cmd_out = subprocess.run(command,
                             stdout=None if args.verbose else subprocess.DEVNULL,
                             stderr=subprocess.STDOUT)
    if cmd_out.returncode != 0:
        print(f"Error while running {command}.")
        exit(-1)

def compile(args):
    GREEN='\033[0;32m'
    NC='\033[0m' # No color

    if args.compile:
        comp_out = subprocess.run(["scons", "-j", str(args.j)], cwd=args.mitsuba_path)
        if comp_out.returncode != 0:
            exit(-1)
        print(f"\n{GREEN}Compilation Successful.{NC}\n")


if __name__ == "__main__":
    main()
