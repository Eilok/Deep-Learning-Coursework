### **Core Logic**
Propose an adaptive quantization framework to quantize models and accelerate the processing speed of existing super-resolution models.

The framework includes two bit-mapping modules:
- Mapping images to image-level bit adaptation factors.
- Obtaining layer-level adaptation factors.

---

### **Methods**
#### **1. Asymmetric Uniform Quantizer**
Address the highly asymmetric distribution of activation values.

#### **2. Method for Accurate Quantizer Design**
- **Observations**:
  1. **Inter-layer relative order consistency**: The relative ranking of layers' sensitivity to quantization remains stable when processing different input images.
  2. **Inter-image relative variation consistency**: Quantization factors vary consistently across different images.
- **Proposed Method**:
  1. **Separated Learning Strategy**: Learn quantization factors separately for images and layers.
  2. **Bitwidth Allocation Formula (Quantizer)**:  
     \[
     b^k_j = b_{\text{base}} + b^I_j + b^L_k
     \]

---

#### **3. Image-to-Bitwidth Mapping Based on Complexity**
- **Image Complexity Calculation**: Use average gradient density to measure image complexity.
- **Bitwidth Factor Mapping Formula**:  
  \[
  b_I^j = I2B(c(I_L^j)) = 
  \begin{cases} 
  -1, & c(I_L^j) < l_{\text{i2b}} \\ 
  +1, & c(I_L^j) > u_{\text{i2b}} \\ 
  0, & \text{otherwise}.
  \end{cases}
  \]  
- **Threshold Determination for Complexity**: Calculate the complexity distribution of a small subset of images to determine thresholds \( l_{\text{i2b}} \) and \( u_{\text{i2b}} \).

---

#### **4. Layer-to-Bitwidth Mapping Based on Sensitivity**
- **Quantization Sensitivity Calculation**: Feed calibration images into the pre-trained model, record activations for each layer, and calculate the **standard deviation** as sensitivity. Higher standard deviation indicates higher sensitivity.
- **Bitwidth Factor Mapping Formula**:  
  \[
  b_L^k = L2B(s^k) = 
  \begin{cases} 
  -1, & s^k < l_{\text{i2b}}, \\ 
  +1, & s^k > u_{\text{i2b}}, \\ 
  0, & \text{otherwise}.
  \end{cases}
  \]  
- **Threshold Determination for Sensitivity**: Collect sensitivity values of all layers and determine thresholds \( l_{\text{i2b}} \) and \( u_{\text{i2b}} \).

*Note: This step can be performed in advance and does not occupy inference time.*

---

#### **5. Bit-Aware Pruning**
Adjust the pruning range dynamically based on different quantization bitwidths.

#### **6. Fine-Tuning**
Further optimize the parameters of the quantized network using the following losses:
- **Pixel-level Reconstruction Loss** \( L_{\text{pix}} \)
- **Layer-level Reconstruction Loss** \( L_{\text{skt}} \)
- **Bitwidth Regularization Loss** \( L_{\text{bit}} \)

---

### **Evaluation Metrics**
1. **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate closer resemblance to high-resolution images.
2. **SSIM (Structural Similarity Index)**: Values closer to 1 indicate better structural preservation.

---

### **Improvements**
To improve the image complexity calculation:
1. Utilize ViT (Vision Transformer) attention maps to evaluate complexity. See `AdaBM/src/model/edsr_vit.py` for details.
2. Apply frequency domain analysis methods. See `AdaBM/src/model/edsr_ft.py` for details.
