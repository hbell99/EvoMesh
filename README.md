# EvoMesh (ICML 2025)
About code release of "EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions", ICML 2025
[[Arxiv]](https://arxiv.org/abs/2410.03779) [[Project Page]](https://hbell99.github.io/evo-mesh/).

Graph neural networks have been a powerful tool for mesh-based physical simulation. To efficiently model large-scale systems, existing methods mainly employ hierarchical graph structures to capture multi-scale node relations. However, these graph hierarchies are **typically manually designed and fixed**, limiting their ability to adapt to the evolving dynamics of complex physical systems. We propose EvoMesh, a fully differentiable framework that **jointly learns graph hierarchies and physical dynamics, adaptively guided by physical inputs.** EvoMesh introduces anisotropic message passing, which enables direction-specific aggregation of dynamic features between nodes within each hierarchy, while simultaneously learning node selection probabilities for the next hierarchical level based on physical context. This design creates more flexible message shortcuts and enhances the model's capacity to capture long-range dependencies. 

<p align="center">
<img src=".\pic\comparison.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Comparison of mesh-based physical simulation models. Dynamic hierarchy refers to hierarchical graph structures that evolve over time. Adaptive indicates that the graph structures are determined by physical inputs. Prop. denotes feature propagation.
</p>


<p align="center">
<img src=".\pic\EvoMesh.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Overview of EvoMesh.
</p>

## Requirements

- Pytorch
- PyG
- Numpy
- h5py
- TensorBoard
- SciPy
- scikit-learn
- sparse-dot-mkl


## Datasets


Please follow the instruction of [MeshGrahNet](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) to download datasets. To convert your TensorFlow dataset (`.tfrecord`) into HDF5 files (`.h5`), modify the paths for `tf_dataset` and `root` in the corresponding config. For example, for the cylinderflow dataset, modify `TF_DATASET_PATH` and `SAVE_ROOT`,  to point to your TensorFlow dataset directory and desired output directory. Then, run the following command:

```bash
python misc/parse_tfrecord.py
```

Please maintain the file structure shown below to run the script by default.

```sh
|───data
│   └───cylinder
|       └───outputs_test
|       └───outputs_train
|       └───outputs_valid
│       │   meta.json
│   └───...
```

## Pretrained Weights

The pretrained weights can be download from this [link](https://drive.google.com/drive/folders/1jGA2M5Vahc_d9WZJ6_Ov6TBZSM_RsBKD?usp=sharing).

## Training

```sh
# ./run.sh $case_name ./configs/$case_name $mode $restart_epoch

# case_name: [cylinder, flag]
# ./configs/$case_name: stores the corresponding config files of a case
# mode: [0:train, 1:local test, 2: global rollout]
# restart_epoch: -1 (or leave blank) to train from the start; 0, 1... to reload the stored ckpts of a certain frame

# e.g. train font from scratch
./run.sh cylinder ./configs/cylinder 0 -1
# e.g. local test RMSE of cylinder at epoch 19
./run.sh cylinder ./configs/cylinder 1 19
# e.g. global rollout RMSE of airfoil at epoch 39
./run.sh flag ./configs/airfoil 2 39
```


## Results

EvoMesh consistently outperforms the compared mod- els across all benchmarks.

<p align="center">
<img src=".\pic\results.png" height = "250" alt="" align=center />
<br><br>
<b>Table 1.</b> Results on four datasets.
</p>


## Showcases
<p align="center">
<img src=".\pic\showcase1.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 3.</b> Prediction showcases over 400 future steps on CylinderFlow.
</p>


## Poster

<div  style="display:flex; flex-direction:row;">
    <figure>
        <img src="./pic/poster.png" height=300px/>
    </figure>
</div>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{
    deng2025evomesh,
    title={EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions},
    author={Huayu Deng and Xiangming Zhu and Yunbo Wang and Xiaokang Yang},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=ZZvTc92dYQ}
}
```

## Credits

The codes refer to the implemention of [BSMS-GNN](https://github.com/Eydcao/BSMS-GNN). Thanks for the authors！