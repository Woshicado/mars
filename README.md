# MARS: Multi-sample Allocation through Russian roulette and Splitting

![Teaser](assets/teaser.jpg)

This repository contains the authors' Mitsuba implementation of "[MARS: Multi-sample Allocation through Russian roulette and Splitting](https://woshicado.eu/MARS)".
We implemented our algorithm in
- [mitsuba/src/integrators/bmars](mitsuba/src/integrators/bmars) for the bidirectional application,
- [mitsuba/src/integrators/path/guided_path.cpp](mitsuba/src/integrators/path/guided_path.cpp) for the guided application, and,
- [mitsuba/src/integrators/effmis](mitsuba/src/integrators/effmis) for the [efficiency-aware MIS variant](https://doi.org/10.1145/3528223.3530126), denoted 'brute-force' in the paper.

In [mitsuba/src/integrators/path/recursive_path.cpp](mitsuba/src/integrators/path/recursive_path.cpp), the authors' version of [EARS: Efficiency-Aware Russian roulette and Splitting](https://graphics.cg.uni-saarland.de/publications/rath-sig2022.html) is contained as well, which is also available at https://github.com/iRath96/ears.


## Running

We provide a python wrapper to call mitsuba with pre-defined configurations: [render.py](render.py). This script automatically parses [config.yaml](config.yaml) to assign the correct values to our implementations to reproduce the results described in the paper. To run it, simply compile and source mitsuba ­– then call
```
render.py --run --scenes cbox [+add. scenes] --budget 300 --render-configs B-MARS G-MARS [+add. configs]
```

We also provide a [dockerfile](Dockerfile), that will automatically download 4 of the scenes we evaluated in the paper from [Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/): `bedroom`, `dining-room`, `kitchen`, and `living-room`. By default, its entrypoint is calling `render.py`. So you can call the docker image in the same way as if you were calling the `render.py` script directly. With the `--outdir` flag, you can redirect the rendered output to a mounted directory in docker.


### Parameters

The rendering script supports [argcomplete](https://pypi.org/project/argcomplete/) to allow for autocompletion. Follow the installation instructions and you should be able to press <kbd>⇥Tab</kbd> for suggestions and completions. (This does not work for running the docker image.)

The most important arguments are given in this table.
For a complete list, please consult the help page with `--help` or the [complete parameter list](#complete-parameter-list):

| short        | long               | args        | action                                                                                  |
| ------------ | ------------------ | ----------- | ----------------------------------------------------------------------------------------|
| -r           | --run              | —           | By default, the script simply prints the commands. To actually run them, pass this flag |
| -S           | --scenes           | `List[str]` | Render's the provided list of scenes                                                    |
| -cfg         | --render-configs   | `List[str]` | _For supported arguments see [pre-defined configurations](#pre-defined-configurations)_ |
| -b           | --budget           | `int`       | Rendering budget in seconds                                                             |


### Pre-defined configurations

The `render-configs` are the names found in [config.yaml](config.yaml) under `strategies`. For ease of use, we list them here as well. Just use the 'Configuration Name' provided below to load and render a particular configuration.
On top of that, we support two additional keywords `paper` and `all`. If contained, they overwrite all other configuration arguments.
| Configuration Name | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| paper              | Activates all configurations that are presented in the _main_ paper |
| all                | Activates all configurations                                        |


#### Bidirectional configurations

| Configuration Name | Description in the paper | Description                                                                                  |
|--------------------|--------------------------|----------------------------------------------------------------------------------------------|
| B-classicRR        | classic RR               | BDPT in the MARS implementation                                                              |
| B-brute-force      | brute-force              | 'brute-force' ([efficiency-aware MIS](https://github.com/pgrit/EfficiencyAwareMIS) variant)  |
| B-EARS             | EARS                     | [EARS](https://github.com/iRath96/ears) (applied to BDPT)                                    |
| B-MARS-bu          | Ours budget-unaware      | MARS (bidirectional budget-UNaware)                                                          |
| B-MARS             | Ours                     | MARS (bidirectional budget-aware)                                                            |
| B-EARS-1LP         | —                        | EARS applied to BDPT and NEE with exactly 1 light path always                                |
| B-MARS-1LP-bu      | —                        | MARS (on BSDF/NEE; exactly 1 LP; bidirectional budget-UNaware)                               |
| B-MARS-1LP         | —                        | MARS (on BSDF/NEE; exactly 1 LP; bidirectional budget-aware)                                 |
| B-bdpt             | —                        | BDPT (with classic throughput-based RR)                                                      |


#### Guiding configurations

| Configuration Name | Description in the paper | Description                                                                                     |
|--------------------|--------------------------|-------------------------------------------------------------------------------------------------|
| G-classicRR        | classic RR               | classic throughput-based path guiding                                                           |
| G-EARS             | EARS                     | [EARS](https://github.com/iRath96/ears) (on BSDF+guiding mixture and NEE) - image KL divergence |
| G-MARS-grad        | Ours + Grad. Descent     | MARS (on BSDF+guiding mixture and NEE budget-aware)                                             |
| G-MARS-bu          | Ours budget-unaware      | MARS (guiding budget-UNaware)                                                                   |
| G-MARS             | Ours                     | MARS (guiding budget-aware)                                                                     |
| G-EARS-var         | EARS variance            | EARS (on BSDF+guiding mixture and NEE) - variance                                               |
| G-EARS-ivar        | EARS image variance      | EARS (on BSDF+guiding mixture and NEE) - image variance                                         |
| G-EARS-kl          | EARS KL divergence       | EARS (on BSDF+guiding mixture and NEE) - KL divergence                                          |
| G-MARS-grad-bu     | —                        | MARS (on BSDF+guiding mixture and NEE budget-UNaware)                                           |
| G-ADRRS            | —                        | path guiding with ADRRS \[UNTESTED\]                                                            |


### Running Manually

Of course you are not restricted to using the provided python wrapper but can instead call mitsuba directly. However, please note that not all argument pairs are compatible with each other, or have not been tested extensively.


## Debugging AOVs

Our integrators support outputting many insightful AOVs (e.g., average splitting factors at each depth, computation cost of each pixel, …).
Since activating them causes overhead, we have disabled outputting them by default.
To enable them, define the preprocessor macro `MARS_INCLUDE_AOVS` either in your scons config or in the respective source file.


## Compilation

To compile the Mitsuba code, please follow the instructions from the [Mitsuba documentation](http://mitsuba-renderer.org/docs.html) (sections 4.1.1 through 4.6). Since our new code uses C++11 features, a slightly more recent compiler and dependencies than reported in the mitsuba documentation may be required. We only support compiling mitsuba with the [scons](https://www.scons.org) build system.

We tested our Mitsuba code on
- Linux (Ubuntu 20.04 and 22.04, `x64`; Fedora 38-40, `x64`)
- macOS (Monterey, `arm64`)


## License

The new code introduced by this project is licensed under the GNU General Public License (Version 3). Please consult the bundled LICENSE file for the full license text.


## Citing MARS

To cite us, you can use the following bibtex entry, citing the official Version of Record published at Siggraph Asia 2024.
```bibtex
@inproceedings{meyerMARSMultisampleAllocation2024,
  title      = {{{MARS}}: {{Multi-sample Allocation}} through {{Russian}} roulette and {{Splitting}}},
  shorttitle = {{{MARS}}},
  booktitle  = {{{SIGGRAPH Asia}} 2024 {{Conference Papers}}},
  author     = {Meyer, Joshua and Rath, Alexander and Yazici, {\"O}mercan and Slusallek, Philipp},
  year       = {2024},
  month      = {nov},
  series     = {{{SA}} '24},
  pages      = {1--10},
  publisher  = {Association for Computing Machinery},
  address    = {Tokyo, Japan},
  doi        = {10.1145/3680528.3687636},
  url        = {https://doi.org/10.1145/3680528.3687636},
  isbn       = {979-8-4007-1131-2/24/12},
  langid     = {english},
  numpages   = {10},
}
```


## Complete-Parameter-List

| short        | long               | args        | action                                                                                               |
| ------------ | ------------------ | ----------- | ---------------------------------------------------------------------------------------------------- |
| -r           | --run              | —           | By default, the script simply prints the commands. To actually run them, pass this flag              |
| -S           | --scenes           | `List[str]` | Render's the provided list of scenes                                                                 |
| -cfg         | --render-configs   | `List[str]` | _For supported arguments [pre-defined configurations](#pre-defined-configurations)_                  |
| -b           | --budget           | `int`       | Rendering budget in seconds                                                                          |
| -c           | --compile          | —           | Compiles mitsuba before running the commands. Does not run any commands if compilation fails         |
| -v           | --verbose          | —           | Prints mitsuba's output to stdout                                                                    |
| —            | --maxDepth         | `int`       | Maximum path depth                                                                                   |
| —            | --rrDepth          | `int`       | Depth at which RR and Splitting starts                                                               |
| -o           | --outdir           | `str`       | Output directory                                                                                     |
| —            | --train-fraction   | `d: int`    | **(GUIDING)** Use `budget/d` seconds for training                                                         |
| —            | --train-budget     | `int`       | **(GUIDING)** Explicit train budget in seconds. Disables train-fraction if given                     |
| —            | --train-iterations | `int`       | **(GUIDING)** Number of training iterations                                                          |
| —            | --mis-exp          | `int`       | **(Bidir. MARS)** Exponent of the MIS power heuristic                                                |
| -s           | --skip             | `n: int`    | Does not run the first `n` commands                                                                  |
| —            | --memLimit         | `int`       | **(EARS+MARS)** Memory limit of cache in MB. This is multiplied by the number of techniques for MARS |
| —            | --partial          | `s: int`    | Passes the `-r` flag to mitsuba, saving the rendered image every `s` seconds                         |
| —            | --store-train      | —           | **(EARS + Bidir. MARS)** Saves the training iterations images next to the output file                |
| —            | --mitsuba-path     | `str`       | Path to the parent directory of mitsuba's source. Necessary for compile flag to work                 |
| —            | --scene-dir        | `str`       | Path to the directory containing the scene folders                                                   |
| —            | --scene-file       | `str`       | Scene file name with extension                                                                       |
| —            | --file-name        | `str`       | Output file name without extension                                                                   |
| —            | --outlierFactor    | `int`       | **(BRUTE-FORCE)** Changes outlier rejection factor. Higher = Less rejection                          |
| —            | --subdir           | `str`       | Subdirectory to put the outputs of this execution's results into                                     |
| -d           | --delete           | —           | Clears the output directory before execution                                                         |
| —            | --disable-colors   | —           | Disables the color output of the printed commands                                                    |
| -g           | --gdb              | —           | Runs the commands using gdb                                                                          |
| —            | --li               | —           | **(NOT THOROUGHLY TESTED)** Activates light-image rendering for BDPT/MARS/effmis                     |
| -j           | —                  | `int`       | Number of cores used by scons for compilation and building of mitsuba                                |
| -h           | --help             | —           | Prints all arguments and their descriptions                                                          |
