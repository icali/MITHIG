# ALEPH data download URL: https://myspace.iii.org.tw/d/f/609047911176649049
# Column Names: entry,nParticle,pwflag,charge,pt,eta,phi,px,py,pz

# We tried two GAN Python packages: tabgan and ctgan.
# Steps: input data -> data pre-processing -> generate data -> data post-processing-> cut out-of-range entries (optional) -> output file into .csv

############
## tabgan ##
############
See [here](https://pypi.org/project/tabgan/#description) for more detail.

### Packages required
* python 3.8.8
* numpy 1.20.1
* pandas 1.2.3
* pip 21.0.1
* scikit-learn 0.24.1
* scipy 1.6.0
* tabgan 1.0.3 (on pypi)

### How to Use Library
* Installation: `pip install tabgan`
* To generate new data to train by sampling and then filtering by adversarial training
  call `GANGenerator().generate_data_pipe`:
* [class GANGenerator source code](https://github.com/Diyago/GAN-for-tabular-data/blob/master/src/tabgan/sampler.py)
* [\_CTGANSynthesizer](https://github.com/Diyago/GAN-for-tabular-data/tree/93d45d2c5a66657a6ba0a627294e9a9cc1fc97a7/src/_ctgan)

``` python
from tabgan.sampler import GANGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# input data
pf = pd.read_csv("LEP1Data1992_recons_aftercut-MERGED.root.20200630.top100kEntries.csv")

# data pre-processing
df.drop("nParticle", axis=1)
drop_cols = ["pt", "eta", "phi"]
for drop_col in drop_cols:
	df = df.drop(drop_col, axis=1)
x = df.loc[:, [x for x in df.columns if x != "entry"]]
y = pd.DataFrame(df.loc[:, 'entry'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
train = x_train
target = y_train
test = x_test

# generate data
new_train1, new_target1 = GANGenerator(cat_cols=['pwflag', 'charge']).generate_data_pipe(train, target, test, )

# data post-processing
new_train1['entry'] = new_target1
new_train1 = new_train1[df.columns]
sorted_new_train1 = new_train1.sort_values(by='entry').reset_index(drop=True)
sorted_new_train1['nParticle'] = sorted_new_train1.groupby(by='entry')['entry'].transform('count')
sorted_new_train1_save = sorted_new_train1[['entry','nParticle','pwflag','charge','px','py','pz']]

# cut out-of-range entries (optional)
sorted_new_train1_save = sorted_new_train1_save.drop(sorted_new_train1_save[sorted_new_train1_save["entry"] < 0].index)
sorted_new_train1_save = sorted_new_train1_save.drop(sorted_new_train1_save[sorted_new_train1_save["entry"] > 100000].index)
sorted_new_train1_save = sorted_new_train1_save.reset_index(drop=True)

# output file into .csv
sorted_new_train1_save.to_csv("GAN_tabgan_10kentry_sorted_cut.csv", index=False)


# Or alternatively use GANGenerator with all params defined
new_train2, new_target2 = GANGenerator(gen_x_times=1.1, cat_cols=['pwflag', 'charge'], bot_filter_quantile=0.001,
                                       top_filter_quantile=0.999,
                                       is_post_process=True,
                                       adversaial_model_params={
                                           "metrics": "AUC", "max_depth": 2,
                                           "max_bin": 100, "n_estimators": 500,
                                           "learning_rate": 0.02, "random_state": 42,
                                       }, pregeneration_frac=2,
                                       epochs=500).generate_data_pipe(train, target,
                                                                      test, deep_copy=True,
                                                                      only_adversarial=False,
                                                                      use_adversarial=True)

```

For more input parameters details, see [here](https://pypi.org/project/tabgan/#description).


###########
## ctgan ##
###########
[CTGAN User Guide](https://sdv.dev/SDV/user_guides/single_table/ctgan.html)
See [here](https://github.com/sdv-dev/CTGAN) for more detail.

### Packages required
* python 3.8.8
* numpy 1.20.1
* pandas 1.1.3
* pip 21.0.1
* pytorch 1.7.0
* scikit-learn 0.24.1
* scipy 1.6.0
* ctgan 0.4.0

### How to Use Library
* Installation: `pip install ctgan` by using pip, `conda install -c sdv-dev -c pytorch -c conda-forge ctgan` by using conda.
* [class CTGANSynthesizer source code](https://github.com/sdv-dev/CTGAN/blob/790c1757ff4e67f50515dee3a16c5f9c5a0ce7cd/ctgan/synthesizers/ctgan.py#L88)

```python
from ctgan import CTGANSynthesizer
import pandas as pd

# input file
df = pd.read_csv('LEP1Data1992_recons_aftercut-MERGED.root.20200630.top100kEntries.csv')

# data pre-processing
categorical_features = ['pwflag', 'charge']
df = df.drop('nParticle', axis=1)
drop_cols = ["pt", "eta", "phi"]
for drop_col in drop_cols:
	df = df.drop(drop_col, axis=1)

# generate data
ctgan = CTGANSynthesizer(verbose=True)
ctgan.fit(df, categorical_features, epochs=50)

samples = ctgan.sample(df.shape[0])

# data post-processing
sorted_samples = samples.sort_values(by="entry").reset_index(drop=True)
sorted_samples["nParticle"] = sorted_samples.groupby(by="entry")["entry"].transform("count")
sorted_samples = sorted_samples[['entry','nParticle','pwflag','charge','px','py','pz']]

# cut out-of-range entries (optional)
sorted_samples = sorted_samples.drop(sorted_samples[sorted_samples["entry"] < 0].index)
sorted_samples = sorted_samples.drop(sorted_samples[sorted_samples["entry"] > 100000].index)
sorted_samples = sorted_samples.reset_index(drop=True)

# output file to .csv
sorted_samples.to_csv("GAN_ctgan_10kentry_sorted_cut.csv", index=False)

```

For more input parameters details, see [here](https://sdv.dev/SDV/user_guides/single_table/ctgan.html#how-to-modify-the-ctgan-hyperparameters).
