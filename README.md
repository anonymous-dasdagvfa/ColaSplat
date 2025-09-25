# CoLaSplat

CoLaSplat implements our core algorithm for compressed scene representation and rendering. The main logic is in [`admm.py`](admm.py).

---

## Installation

Clone the repository and initialize submodules:

```bash
git clone <your-repo-url>
cd <your-project-directory>
git submodule update --init --recursive
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate colasplat
```


## Quick Start
Use the demo compressed model to render:

```bash
python render_admm_quant.py -m output_admm_quant/bed --dataset 3dovs --include_feature
```





---

## Dataset

Please refer to the following repositories for the datasets:

* [3D-OVS](https://github.com/Kunhao-Liu/3D-OVS)
* [LERF](https://github.com/minghanqin/LangSplat)

The data should be organized as follows (example for the `bed` scene from 3D-OVS):

```
data/
└── 3dovs/
    └── bed/
        ├── images/
        │   ├── img_0001.png
        │   ├── img_0002.png
        │   └── ...
        └── annotations/
            ├── ann_0001.json
            ├── ann_0002.json
            └── ...
```

We provide a demo of a compressed model trained on the `bed` scene. The point clouds and codebook files can be found at:

```
output_admm_quant/bed/point_cloud/iteration_10000
```

For data preprocessing, please refer to the [LangSplat repository](https://github.com/minghanqin/LangSplat).

---


The compressed results will be saved in:

```
output_admm_quant/
```

## Training process

### 1. Generate initial 3DGS point cloud

For the `bed` scene, the initial point cloud should be placed at:

```
CoLaSplat/data/3dovs/bed/output/bed/point_cloud/iteration_30000/point_cloud.ply
```

### 2. Semantic learning

Generate it using the provided script:

```bash
CoLaSplat/scripts/3dovs.sh
```

This corresponds to the first 30,000 iterations in the paper.

---

### 3. Compression

After generating the initial point cloud, run the compression:

```bash
CoLaSplat/scripts/3dovs_admm_quant.sh
```

Results will be saved at:

```
CoLaSplat/output_admm_quant/bed/train
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
