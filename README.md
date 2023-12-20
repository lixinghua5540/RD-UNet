# RD-UNet
Title: *Residual Dual U-shape Networks With Improved Skip Connections for Cloud Detection* [[paper]](https://ieeexplore.ieee.org/document/10335738/)<br>

A. Li, X. Li and X. Ma, "Residual Dual U-shape Networks With Improved Skip Connections for Cloud Detection," IEEE Geoscience and Remote Sensing Letters, IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1-5, Art. no 5000205, 2024.
<br>
<br>
***Introduction***<br>
<br>
Cloud detection in remote sensing images is a challenging task that plays a crucial role in various applications. A novel residual dual U-shape networks (RD-UNet) is proposed for cloud detection. The primary innovation lies in that it cascades two U-shaped networks and leverages a residual-like connection to enhance information flow between two networks, thus optimizing details and edge information. Moreover, another contribution is that the improved skip connections (ISC) efficiently facilitate multiscale feature utilization, aiding in the identification of thin clouds and distinguishing other confounding land features. The effectiveness of RD-UNet was demonstrated through extensive experiments on two public datasets, outperforming state-of-the-art methods with a nearly 2% improvement in F1 score and superior visual effect for multispectral images.
<br>
<br>
![image](https://github.com/lixinghua5540/RD-UNet/blob/master/RD-UNet/images/Fig.%201.png)
<br>Fig. 1.The architecture of the proposed residual dual U-shape networks (RD-UNet).<br>
<br>
<br>
***Usage***
<br>
tensorflow<br>
run train.py
