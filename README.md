# A Spatial Semantics and Continuity Perception Attention for Remote Sensing Water Body Change Detection ([ArXiv](https://arxiv.org/abs/2511.16143v1))

## The HSRW-CD Dataset

The HSRW-CD dataset is the first high spatial resolution dataset
dedicated to WBCD (Water Body Change Detection). Specifically, it comprises 2,085 image pairs spanning diverse geographical regions and environmental contexts, encompassing various water body types such as urban waterways, river systems, lacustrine environments, and artificial reservoirs. These image pairs containing Red (R), Green (G), and Blue (B) bands are from more than 7 cities or districts in China, including Beijing, Chongqing, Chengdu, Hangzhou, Wuhan, Shenzhen, and Shanghai, covering four major geographic regions of North China, East China, South China and Southwest China. The dataset spans temperate monsoon and subtropical monsoon climate zones, as well as diverse landform types including plains, mountainous areas, hills and dense water network regions, which ensures the samples cover diverse natural and artificial water body scenarios in different geographic environments, with sufficient representativeness and generalization potential.  

The bi-temporal images are manually collected from three high-quality data sources: Jilin-1 series commercial high-resolution optical remote sensing satellites (GSD < 0.75 m), the Google Earth platform (2 m GSD), and multi-platform aerial sensors (0.5 m–3 m GSD). The time interval between bi-temporal images ranges from 6 months to 3 years according to inter-annual hydrological variations, artificial water body regulation, and urban water system evolution. Based on this interval, we carefully screen and retain images with sufficient and reasonable water body changes, thereby ensuring the quality, validity and diversity of the dataset. We then perform standardized preprocessing on all selected images, including geometric fine registration with an error of less than 1 pixel, radiometric normalization, and atmospheric correction, to eliminate systematic errors between bi-temporal images. The annotation of the dataset is carried out by an expert group of Earth vision applications, which guarantees high label accuracy. A double-blind cross-check mechanism is adopted for quality control: two groups of experts independently verify the annotation results, and inconsistent regions are jointly reviewed and revised to ensure the final annotation accuracy is higher than 98\%, with unqualified samples eliminated from the dataset. In addition, all bi-temporal images are 512 $\times$ 512 pixels in size. The image dataset is divided into training, validation, and test subsets via stratified random sampling based on city distribution and water body types at a 7:1:2 ratio, yielding 1,476, 203, and 406 independent image pairs for each subset.

Obtain it by:
[Baidu Netdisk-Link]
(https://pan.baidu.com/s/1wygwa15uOreD3-z_MT3wPw?pwd=opac)

## Pre-trained Model Weights and Log Files

Obtain these by:
[Quark Netdisk-Link]
(https://pan.quark.cn/s/d72ceb2158b3?pwd=5TUU)
