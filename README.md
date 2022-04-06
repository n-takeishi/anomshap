# On anomaly interpretation via Shapley values

Codes for the method presented in the following paper:

Naoya Takeishi and Yoshinobu Kawahara, [On anomaly interpretation via Shapley values](https://arxiv.org/abs/2004.04464), arXiv:2004.04464, 2020.

## Usage

First, download raw data files (.mat) from the ODDS (http://odds.cs.stonybrook.edu/), such as:

- Thyroid, http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
- Musk, http://odds.cs.stonybrook.edu/musk-dataset/
- WBC, http://odds.cs.stonybrook.edu/wbc/
- BreastW, http://odds.cs.stonybrook.edu/breast-cancer-wisconsin-original-dataset/
- Arrhythmia, http://odds.cs.stonybrook.edu/arrhythmia-dataset/

Save the downloaded .mat files in `data/raw` directory.

For instruction purpose, suppose you have downloaded [the Thyroid dataset](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/) and saved the `thyroid.mat` file in `data/raw` directory.

Then, execute `build_datasets.m` by Octave or MATLAB, which will produce formatted data in `data/features/thyroid` directory.

Preparing dataset like this, execute the notebook `demo_synthetic_anomaly.ipynb` to demonstrate the proposed method (along with two baseline methods).

## Author

Naoya Takeishi

https://ntake.jp/
