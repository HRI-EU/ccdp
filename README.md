# CCDP

# CCDP Internal Repository  

This repository contains the codebase for the paper **[CCDP](https://hri-eu.github.io/ccdp/)**. The code was developed by the CCDP team during [A. Razmjoo](https://amirrazmjoo.github.io/)'s internship at HRI (Oct. 2024 – Mar. 2025).  

If you encounter any issues or need clarification, please contact A. Razmjoo directly or reach out to other team members.  

## Repository Structure  

This repository is organized around the cleaned door-manipulation pipeline and the assets it depends on.  

- **Main Codes/**: Cleaned and structured door pipeline used for the paper results.  
  - `door_cleaned.ipynb`: Jupyter notebook version of the door example.  
  - `door_pipeline_clean.py`: Scripted pipeline with a `main()` entrypoint for data generation, training, and sampling.  
  - `door_config.py`: Centralized configuration, paths, and defaults.  
  - `door_models.py`: Model blocks, datasets, and diffusion helpers.  
- **models/**: Pretrained checkpoints and cached datasets (`*.pth`, `*.pkl`, `*.pt`).  
- **xml/**: Mujoco environment definitions and mesh assets used by the door task.  
- `ccdp.yml`: Conda environment specification.  
- `requirements.txt`: Pip dependencies if you are not using conda.  


## Third-Party Assets

This repository includes third-party MuJoCo models and assets. See `THIRD_PARTY_NOTICES.md` for required attributions and license texts.


## Usage  

To use this repository, follow these steps:  

1. Install the required dependencies:  
   ```bash
   conda env create -f ccdp.yml
   ```
   Or, with pip:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the pretrained models and cached demos (required for the demos and to reproduce results):
   ```bash
   python download_models.py
   ```
3. **Run the notebook:**
   - Notebook:
     ```bash
     jupyter notebook "Main Codes/door_cleaned.ipynb"
     ```
   
The scripted pipeline caches generated demos and checkpoints in `models/`.  

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{Razmjoo25IROS,
	author={Razmjoo, A. and Calinon, S. and Gienger, M. and Zhang, F.},
	title={{CCDP}: Composition of Conditional Diffusion Policies with Guided Sampling},
	booktitle={Proc.\ {IEEE/RSJ} Intl Conf.\ on Intelligent Robots and Systems ({IROS})},
	pages={20036--20043},
	year={2025}
}
```
