# DECUN -- Deep Convergent Unrolling for Non-blind deblurring

This repo contains the implementation of DECUN and some useful files to reuse its building blocks.

## Structure of the code 

1. [data](data) contains data samples for testing. The full data can be download from [Here](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/hzz5333_psu_edu/EgnyDkZ9ZrZHqwz5pfPpITEBrqc8QDONBU4iXWMDCZdaOw?e=SNVsU4).

2. [trained_models](trained_models) contains trained for testing.

4. [test.py](test.py) is the main test file to be run.

4. [option.py](option.py) contains the running option for test.py .

4. [decovNet.py](decovNet.py) contains the DECUN network .




## Setting up the environment

All required packages are found in [requirements.txt](requirements.txt).

* Creating the conda env for "DECUN"
```
conda create env -n "DECUN" python=3.9
```

* Activate "DECUN" conda env
``` 
conda activate DECUN
```

* Install PyTorch
``` 
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

* Install required library
```
pip install -r requirements.txt
```

## Running the Test file.
For more running option can see in the [option.py](option.py)
```
python test.py
```

## If you want to use this code, please cite our paper as
@ARTICLE{Yanan24,
    author = {Yanan Zhao and Yuelong Li and Haichuan Zhang and Vishal Monga and Yonina C. Eldar},
		title = {Deep, convergent, unrolled half-quadratic splitting for image deconvolution},
		journal={arXiv preprint arXiv:2402.12872},
		year = {2024},
}
