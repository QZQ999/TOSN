# Data sources and processing

The industrial observations were obtained from public research collections. The controlled changes in the ETT study are numerical transformations applied to observed time series. This document identifies the source files and describes the processing that turns a series into forecasting examples.

## ETT experiments

ETTh1 and ETTm1 come from the [official ETTDataset repository](https://github.com/zhouhaoyi/ETDataset), at revision `1d16c8f4f943005d613b5bc962e9eeb06058cf07`. The inputs are `ETT-small/ETTh1.csv` and `ETT-small/ETTm1.csv`. ETTh1 contains 17,420 rows, and ETTm1 contains 69,680. Both use the seven measured variables in their stored order. The date column identifies time.

The helper in `reproduction/reproduce.py` downloads this revision and checks its contents against `reproduction/data_sources.json`. The recorded Windows files and the downloaded files differ in line endings. Content verification normalizes those line endings, while preserving both checksums in the source record.

Rows retain their chronological order. The code computes the training length with Python's `int(0.7*N)` and the validation length with `int(0.1*N)`. The remaining rows form the test partition. The exact integer boundaries are stored in each input record. For each variable, the mean and population standard deviation are fitted to training rows. These statistics are then applied to the entire series. The training code replaces a standard deviation below `1e-6` with one.

The observed history has 96 steps. The repeated R0 study uses prediction lengths of 96, 192, 336 and 720 steps. Each target window lies wholly inside its assigned partition. Validation and test histories can include preceding observations. At a prediction length of 96, ETTh1 has 3,389 test windows and ETTm1 has 13,841. The complete counts are recorded in `artifacts/public_dataset_manifest.json`.

## Boiler source

The Boiler collection is published under a CC0 licence at [DOI 10.6084/m9.figshare.28868849](https://doi.org/10.6084/m9.figshare.28868849). The [source publication](https://www.nature.com/articles/s41597-025-05096-4) describes 86,400 observations of 30 variables, sampled every five seconds. The file `xinan_uncompleted_data.csv` retains missing observations. The file `xinan_completed_data.csv` supplies completed values, with the treatment implemented in `AutoReg.py`.

These file versions represent different preprocessing choices. A new run should record the version selected and any further treatment. The manuscript specifies chronological 70%/10%/20% partitions and training-set standardization. The retained original experiment materials leave the selected Boiler version and subsequent treatment unconfirmed.

The DOI and the public release metadata were accessible during the source investigation on 25 October 2026. File requests from the current environment returned HTTP 403. The source publication describes public distribution. Its corresponding-author contact can be used to ask the provider about persistent download problems.

## IoT source and an executable input example

The IoT observations come from the SensorData collection described in [UTSD](https://huggingface.co/datasets/thuml/UTSD). The authors also provide a [NumPy release on Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/). Both routes are listed by the [official project](https://github.com/thuml/Large-Time-Series-Model).

The cloud archive is `UTSD-full-npy.zip`. Its `UTSD-full-npy/IoT/baian/` directory contains 51 data files. We verified the actual contents of `0.npy`: 180,140 rows and 18 columns, stored as `float64`, with finite values throughout. Its SHA-256 checksum is `8af624b260fa4f4b6044ea08073b1023458e33127ed04591c3d8ad6cb8e73ee6`.

The supplied input example uses all rows and all 18 columns of this verified file. Column order is preserved. The CSV headings `v01` through `v18` identify column positions. The conversion keeps the supplied values and row order. Run the following command from the repository root after downloading and extracting the source file:

```text
python reproduction/prepare_iot_example.py --file UTSD-full-npy/IoT/baian/0.npy
```

The command produces `external/industrial_examples/iot_0.csv` and an adjacent `input_record.json`. The record includes the source checksum and the CSV checksum. Row intervals use a zero-based index with an exclusive upper boundary. Training uses `[0,126097)`, validation uses `[126097,144111)`, and testing uses `[144111,180140)`. These boundaries follow the recorded Python implementation, including its floating-point calculation before conversion to an integer. Standardization is performed by the training runner using the training rows. The conversion itself leaves the measurement values intact.

This is a documented source example for a new run. The specific source file or files selected for the original IoT result table remain unconfirmed in the retained records. The example establishes an accessible 18-variable input and an exact preparation procedure. Its source selection is stated separately from the original table.

## Observed measurements and controlled changes

The Boiler and IoT collections contain recorded equipment measurements. The ETT robustness study introduces specified changes at the test boundary while retaining the training and validation series. A level change shifts values relative to each variable's training standard deviation. An amplitude change rescales deviations from the training mean. A gradual change increases the additive shift over the test interval. The transformed series supplies both histories and targets according to their time indices. The dated analysis scripts retain the exact formulas and magnitudes used in those comparisons.
