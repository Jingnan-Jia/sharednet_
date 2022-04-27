# How to use the code?

## Where is the hyper parameters?
### `set_args.py` set general parameters for all datasets.
- model_names
- mode              = 'train'
- infer_data_dir    = 'Not_sure'
- infer_weights_fpath
- infer_ID          = 198
- loss              = 'dice'
- cond_flag         = True
- cond_method       = 'concat'
- cond_pos          = 'enc'
- same_mask_value   = False
- base              = 32
- steps             = 100001
- valid_period
- lr                = 1e-4
- weight_decay      = 1e-4
- cache             = True
- batch_size        = 1
- pps               = 10

### `path.py` set pathes

### `nets.py` set network parameters
- act = "LeakyReLU", "negative_slope": 0.1,
- norm = "batch",
- dropout = 0.1,

### `run.py` 
training accumulate_loss period = 200

### `dataset.py` set target spacing (`tsp`) and patch size (`psz`)
- workers = 12
for lobe/lung:
- tsp: str = "1.5_2.5"
- psz: str = "144_96"

for vessel/AV:
- tsp=None
- psz="144_96"

for liver:
- tsp="1.5_1"
- psz="144_96"
                 
for pancreas:
- tsp="1.5_1"
- psz="144_96"


### `trans.py` 
mask_value_original:
- 'lobe_ru': 1,
- 'lobe_rm': 2,
- 'lobe_rl': 3,
- 'lobe_lu': 4,
- 'lobe_ll': 5,
- 'lobe_all': [1, 2, 3, 4, 5],
- 'lung': 'positive',
- 'AV_artery': 1,
- 'AV_vein': 2,
- 'AV_all': [1, 2],
- 'vessel': 'positive',  # label='MergePositiveLabels',
- 'liver': 1,
- 'pancreas': 1

mask_value_target:
- 'lobe_ru': 1,
- 'lobe_rm': 2,
- 'lobe_rl': 3,
- 'lobe_lu': 4,
- 'lobe_ll': 5,
- 'lobe_all': 6,
- 'lung': 7,
- 'AV_artery': 8,
- 'AV_vein': 9,
- 'AV_all': 10,
- 'vessel': 11,  # label='MergePositiveLabels',
- 'liver': 12,
- 'pancreas': 13

Others:
- Clip intensity to -1000, 1000.
- RandAffined
- RandGaussianNoised

## how does the code work?
1. `record_1st` records all the hyper parameters seted by `set_args.py`.
2. run code
3. `record_2nd` records best metrics end time.


