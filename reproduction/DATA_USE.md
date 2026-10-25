# Data sources and processing

The industrial observations were obtained from public research collections. The controlled changes in the ETT study are numerical transformations applied to observed time series. This document identifies the source files and describes the processing that turns a series into forecasting examples.

## ETT experiments

ETTh1 and ETTm1 come from the [official ETTDataset repository](https://github.com/zhouhaoyi/ETDataset), at revision `1d16c8f4f943005d613b5bc962e9eeb06058cf07`. The inputs are `ETT-small/ETTh1.csv` and `ETT-small/ETTm1.csv`. ETTh1 contains 17,420 rows, and ETTm1 contains 69,680. Both use the seven measured variables in their stored order. The date column identifies time.

The helper in `reproduction/reproduce.py` downloads this revision and checks its contents against `reproduction/data_sources.json`. The recorded Windows files and the downloaded files differ in line endings. Content verification normalizes those line endings, while preserving both checksums in the source record.

Rows retain their chronological order. The code computes the training length with Python's `int(0.7*N)` and the validation length with `int(0.1*N)`. The remaining rows form the test partition. The exact integer boundaries are stored in each input record. For each variable, the mean and population standard deviation are fitted to training rows. These statistics are then applied to the entire series. The training code replaces a standard deviation below `1e-6` with one.

The observed history has 96 steps. The repeated R0 study uses prediction lengths of 96, 192, 336 and 720 steps. Each target window lies wholly inside its assigned partition. Validation and test histories can include preceding observations. At a prediction length of 96, ETTh1 has 3,389 test windows and ETTm1 has 13,841. The complete counts are recorded in `artifacts/public_dataset_manifest.json`.

## Boiler source

The Boiler collection is published under a CC0 licence at [DOI 10.6084/m9.figshare.28868849](https://doi.org/10.6084/m9.figshare.28868849). The [source publication](https://www.nature.com/articles/s41597-025-05096-4) describes 86,400 observations of 30 variables, sampled every five seconds. The file `xinan_uncompleted_data.csv` retains missing observations. The file `xinan_completed_data.csv` supplies completed values, with the treatment implemented in `AutoReg.py`.

The released reconstruction selects `xinan_completed_data.csv`, using the complete 86,400-row sequence and all 30 measurement columns in their supplied order. This choice adopts the provider's documented completion procedure. The preparation preserves those completed values. It checks that all measurements are finite and stops for inspection if that check fails. It preserves the stored row order and sampling interval.

After downloading the completed CSV, prepare it with the following command:

```text
python reproduction/prepare_boiler_input.py --file xinan_completed_data.csv
```

The helper expects all 30 measurement columns. If the supplied CSV also includes a separate timestamp or exported row identifier, pass its exact heading through `--metadata-column`. That option excludes only the named metadata field; the helper still requires 30 measured variables and 86,400 rows. The input record saves the actual source headings and the excluded metadata headings. This makes the treatment of CSV metadata visible without silently dropping a measured variable.

The output is `external/boiler_reconstruction/boiler_completed.csv`, accompanied by its input record and a runnable configuration. The CSV adds `row_index` as an ordering identifier, which the forecasting loader removes before forming model inputs. Under the released code's Python calculation, the training interval is `[0,60479)`, the validation interval is `[60479,69119)`, and the test interval is `[69119,86400)`. The upper endpoint is excluded. The file checksum is recorded from the actual downloaded CSV when preparation runs. The helper leaves the provider's values unchanged; normalization is fitted by the training runner.

The DOI and public release metadata were accessible during the source investigation on 25 October 2026. File requests from this environment returned HTTP 403, so the actual Boiler CSV has yet to pass the supplied loader check here. The source publication describes public distribution. Its corresponding author, Zhong Cao (`zhongc@gzhu.edu.cn`), is the provider contact for assistance with a persistent download problem.

## IoT source and the specified input

The IoT observations come from the SensorData collection described in [UTSD](https://huggingface.co/datasets/thuml/UTSD). The authors also provide a [NumPy release on Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/). Both routes are listed by the [official project](https://github.com/thuml/Large-Time-Series-Model).

The cloud archive is `UTSD-full-npy.zip`. Its `UTSD-full-npy/IoT/baian/` directory contains 51 data files. We verified the actual contents of `0.npy`: 180,140 rows and 18 columns, stored as `float64`, with finite values throughout. Its SHA-256 checksum is `8af624b260fa4f4b6044ea08073b1023458e33127ed04591c3d8ad6cb8e73ee6`.

The released reconstruction selects this whole file, using all rows and all 18 measurement columns in their stored order. This defines the interval directly and avoids an additional selection of sensors or time periods. The CSV headings `v01` through `v18` identify the supplied column positions. An additional `row_index` column records the original row number, beginning at zero. The forecasting loader removes that column before using the measured variables. Run the following command from the repository root after downloading and extracting the source file:

```text
python reproduction/prepare_iot_example.py --file UTSD-full-npy/IoT/baian/0.npy
```

The command produces `external/industrial_examples/iot_0.csv` and an adjacent `input_record.json`. It also creates `iot_relu_iTransformer_h96_s2026.json`. The record includes the source checksum and the CSV checksum. Row intervals use a zero-based index with an exclusive upper boundary. Training uses `[0,126097)`, validation uses `[126097,144111)`, and testing uses `[144111,180140)`. These boundaries follow the recorded Python implementation, including its floating-point calculation before conversion to an integer. Standardization is performed by the training runner using the training rows. The conversion itself leaves the measurement values intact.

The actual downloaded IoT input was checked through the forecasting data loader. All 18 measurement columns agree with the source after conversion to the `float32` precision used in training. At prediction length 96, this input produces 125,906 training windows and 17,919 validation windows, followed by 35,934 test windows. The first and last target windows were also checked at prediction lengths 192, 336 and 720. These checks concern input preparation and partitioning.

## Running the specified industrial inputs

After installing the project and the pinned forecasting-model dependency as described in the README, run the generated configuration from the repository root:

```text
python -m ugma.cli run --config external/industrial_examples/iot_relu_iTransformer_h96_s2026.json
python -m ugma.cli run --config external/boiler_reconstruction/boiler_relu_iTransformer_h96_s2026.json
```

Run each command after preparing its corresponding data. Each configuration uses a 96-step history and a 96-step prediction length, with seed 2026. The forecasting model is iTransformer, and the estimated-error module uses the recorded ReLU configuration. Training settings are copied from `relu_ETTh1_iTransformer_h96_s2026` in the run catalogue. The preparation changes the input path and row-identifier setting, then records the actual series length and number of variables. It also gives the fresh run its own output directory. All other model and training settings match that identified template. The generated paths refer to the machine used for preparation, so regenerate them after moving the release.

The original industrial tables retain their reported single-run results. The file choices above specify fresh runs of the released reconstruction. Their preparation records document those choices. The original records identifying the selected files and rows, together with the saved industrial models, are absent from the retained material. Input preparation supplies a training configuration, while prediction accuracy is obtained by running the experiment.

## Observed measurements and controlled changes

The Boiler and IoT collections contain recorded equipment measurements. The ETT robustness study introduces specified changes at the test boundary while retaining the training and validation series. A level change shifts values relative to each variable's training standard deviation. An amplitude change rescales deviations from the training mean. A gradual change increases the additive shift over the test interval. The transformed series supplies both histories and targets according to their time indices. The dated analysis scripts retain the exact formulas and magnitudes used in those comparisons.
