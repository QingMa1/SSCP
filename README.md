# A Spatial Semantics and Continuity Perception Attention for Remote Sensing Water Body Change Detection ([Pattern Recognition](https://doi.org/10.1016/j.patcog.2026.114925))

## 📊The Proposed HSRW-CD
Within the scope of our current knowledge, we construct the first high spatial resolution, large-volume, and application-oriented dataset HSRW-CD for remote sensing Water Body Change Detection (WBCD), featuring imagery with a spatial resolution finer than 3 m, a collection of 2,085 bi-temporal image pairs, and various water body types covering diverse geographic regions. Specifically, all bi-temporal images are 512 $\times$ 512 pixels in size. The image dataset is divided into training, validation, and test subsets via stratified random sampling based on city distribution and water body types at a 7:1:2 ratio, yielding 1,476, 203, and 406 independent image pairs for each subset. Due to the properties of the HSRW-CD dataset, it will further enhance the application of WBCD in sophisticated water resource management.

Obtain it by:
[Baidu Netdisk Link](https://pan.baidu.com/s/1wygwa15uOreD3-z_MT3wPw?pwd=opac)

## 💡SSCP Attention

Focusing on the visual similarity between temporary water bodies and moisture-rich vegetation, as well as the structural continuity of water bodies, we propose a task-specific SSCP attention module for WBCD. Different from conventional attention mechanisms that mainly focus on screening important feature information by recalibrating feature weights to judge which features contribute to prediction, SSCP introduces a semantic-structural collaborative refinement paradigm, which explicitly models the spatial organization rules of water body features through cross-scale semantic extraction and global structural enhancement. As a plug-and-play module, SSCP can be seamlessly integrated into mainstream CNN- and Transformer-based frameworks.

## 📦Pre-trained Weights and Logs

Obtain these by:
[Quark Netdisk Link](https://pan.quark.cn/s/d72ceb2158b3?pwd=5TUU)

## 🔥News
2025.11.20: The original manuscript has been posted on [arXiv](https://arxiv.org/abs/2511.16143).

2026.09.13: Our work is accepted by Pattern Recognition.

2026.09.29: [The fianal version](https://doi.org/10.1016/j.patcog.2026.114925) is available online.

## 📖Citation
If you find this work useful for your research, please feel free to cite it.
```bibtex
@article{ma2026spatial,
  title={A spatial semantics and continuity perception attention for remote sensing water body change detection},
  author={Ma, Quanqing and Chen, Jiaen and Wang, Peng and Zheng, Yao and Zhao, Qingzhan and Zheng, Yuchen},
  journal={Pattern Recognition},
  pages={114925},
  year={2026},
  publisher={Elsevier}
}
