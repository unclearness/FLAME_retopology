# FLAME retopology

A pipeline to align FLAME onto various shapes on your template mesh

- Step 0: Manually align your neutral/canonical mesh to FLAME's one

- Step 1: Compute correspondences

- Step 2: Retopology your meshes with different shapes and expressions into FLAME by optimizing FLAME parameters with PyTorch

## iPhone mesh sample

- `git submodule update --init`
  - FLAME_PyTorch is registered as git submodule. Slightly modified version is used to avoid crash.
- Download FLAME model and extract to `./model/`
  - 2020 version is recommended
- Download addtional landmark info used for RingNet from [here](https://github.com/soubhiksanyal/RingNet/tree/master/flame_model) and put to `./model/FLAME2019/`
- Generate neutral FLAME as obj
  - `python save_FLAME_neutral_obj.py`
- Donwload sample iphone data from [here](https://drive.google.com/file/d/1pRl2M82FbIoPiatFdGricVgoimhUetAe/view?usp=share_link) and extract to `./data/`
  - It includes manually aligned iPhone mesh onto FLAME neutral mesh, which was done by Wrap3. Other sample inputs are also included.
- Compute correspondences between neutral meshes of FLAME and mannually aligned iPhone
  - `python correspondence.py`
- Perform retopogy from iPhone to FLAME
  - `python retopology.py`
