# Medical VQA: MixUp helps keep it simple

This code is the implementation of our paper *Medical VQA: MixUp helps keep it simple*.

* Use the requirements files to setup the enviroment.

* use the following commands to run the experiments. Each file contains its own set of experiments. Change `EXP_NO` variable in the respective .py file to run different experiments. 
```python
python mvqag/experiments/abnormality/2020.py
```

```python
python mvqag/experiments/abnormality/2021.py
```

Our proposed `VQAMixup` code is in `mvqag/train/mixup.py`: function `mixup_data_vqa` and `mixup_criterion_vqa`