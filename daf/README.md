# DAF code

Train unsupervised DAF and run DAF regression on a sample of ABCD (source) and HCP (target). 

## Set up and activate conda environment

```
conda env create -f daf_env.yml
conda activate daf_env
```

*Note: if torch does not work, install it using `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

## Access conda environment in JupyterLab/Jupyter notebook (optional)

```
conda install ipykernel
python -m ipykernel install --user --name daf_env --display-name "daf_env"
```

(refresh page if not seeing "daf_env" in notebook kernel)

## Train unsupervised DAF

Section *"Unsupervised DAF"* in `DAF.ipynb`.

*Note: pretrained DAF model: `saved_model/unsup_daf/G_A2B.pth` and `saved_model/unsup_daf/G_B2A.pth`*

## DAF regression

Section *"DAF regression"* in `DAF.ipynb`.

## References

Code is adapted from https://github.com/aitorzip/PyTorch-CycleGAN



