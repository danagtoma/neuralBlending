# Neural Distance Fields <!-- omit from toc -->

Run script using  `python -m <folder.script>` (no `.py` at the end)

## Tabel of Contents <!-- omit from toc -->
- [Directory Structure](#directory-structure)
  - [Reconstructed 3D meshes (unsigned distance field with Lipschitz loss)](#reconstructed-3d-meshes-unsigned-distance-field-with-lipschitz-loss)
  - [Reconstructed 2D meshes (unsigned distance field with Lipschitz loss)](#reconstructed-2d-meshes-unsigned-distance-field-with-lipschitz-loss)
- [Training Unsigned Distance Fields](#training-unsigned-distance-fields)
    - [1. Trained with Softmax activation on the final layer(level: 0.2)](#1-trained-with-softmax-activation-on-the-final-layerlevel-02)
    - [1.1 Trained with SIREN architecture (level: 0.2)](#11-trained-with-siren-architecture-level-02)
    - [2. Trained with normalization formula on final layer and Signed distance fields as input](#2-trained-with-normalization-formula-on-final-layer-and-signed-distance-fields-as-input)
      - [Using ReLU (level: 0.05)](#using-relu-level-005)
      - [Using Sigmoid (level: 0.2)](#using-sigmoid-level-02)
    - [3. Trained with normalization formula on final layer and Unsigned distance fields as input](#3-trained-with-normalization-formula-on-final-layer-and-unsigned-distance-fields-as-input)
      - [Using ReLU (level: 0.2)](#using-relu-level-02)
      - [Using Sigmoid (level: 0.2)](#using-sigmoid-level-02-1)
        - [500 epochs and 10000 points](#500-epochs-and-10000-points)
        - [100 epochs and 100000 points](#100-epochs-and-100000-points)
- [Training Signed Distance Fields](#training-signed-distance-fields)
    - [1. Trained with Softmax activation on the final layer](#1-trained-with-softmax-activation-on-the-final-layer)
    - [2. Trained with normalization formula on final layer using Sigmoid](#2-trained-with-normalization-formula-on-final-layer-using-sigmoid)
    - [3. Trained with normalization formula on final layer using Softmax and loss function](#3-trained-with-normalization-formula-on-final-layer-using-softmax-and-loss-function)
        - [50 anchors](#50-anchors)
        - [15 anchors](#15-anchors)
- [Experiments for better contours + loss function](#experiments-for-better-contours--loss-function)
  - [U mesh](#u-mesh)
    - [A. Changing number of anchors and epochs](#a-changing-number-of-anchors-and-epochs)
      - [1. 15 anchors](#1-15-anchors)
      - [2. 50 anchors](#2-50-anchors)
      - [3. 200 anchors](#3-200-anchors)
    - [B. Changing the influence factor of the loss (lm)](#b-changing-the-influence-factor-of-the-loss-lm)
    - [C. Best result (50 anchors; 300 epochs) with change in lm](#c-best-result-50-anchors-300-epochs-with-change-in-lm)
  - [Dolphin mesh](#dolphin-mesh)
    - [A. Changing number of anchors and epochs](#a-changing-number-of-anchors-and-epochs-1)
      - [1. 50 anchors](#1-50-anchors)
      - [2. 100 anchors](#2-100-anchors)
      - [3. 200 anchors](#3-200-anchors-1)
    - [B. Changing the influence factor of the loss (lm)](#b-changing-the-influence-factor-of-the-loss-lm-1)
      - [1. 100 epochs](#1-100-epochs)
      - [2. 300 epochs](#2-300-epochs)
      - [3. 500 epochs](#3-500-epochs)
    - [C. Best result](#c-best-result)
  - [Horse mesh](#horse-mesh)
    - [A. Changing number of anchors and epochs](#a-changing-number-of-anchors-and-epochs-2)
      - [1. 300 anchors](#1-300-anchors)
      - [2. 500 anchors](#2-500-anchors)
    - [B. Changing the influence factor of the loss (lm)](#b-changing-the-influence-factor-of-the-loss-lm-2)
      - [1. 300 epochs](#1-300-epochs)
  - [Dragon mesh](#dragon-mesh)
    - [A. Changing number of anchors and epochs](#a-changing-number-of-anchors-and-epochs-3)
      - [1. 100 anchors](#1-100-anchors)
      - [2. 300 anchors](#2-300-anchors)
      - [3. 500 anchors](#3-500-anchors)
    - [B. Changing the influence factor of the loss (lm)](#b-changing-the-influence-factor-of-the-loss-lm-3)
      - [1. 300 anchors](#1-300-anchors-1)
      - [2. 500 anchors](#2-500-anchors-1)
  - [Comparison](#comparison)
  - [Experiments](#experiments)
- [Experiments with sigmoid activation function](#experiments-with-sigmoid-activation-function)
- [Experiments with sin activation function](#experiments-with-sin-activation-function)
    - [A. Diferent architecture dimensions](#a-diferent-architecture-dimensions)
    - [B. Different values for w0](#b-different-values-for-w0)
    - [C. Different number of anchors](#c-different-number-of-anchors)
    - [D. Lower lambda factor](#d-lower-lambda-factor)
    - [E. Best result](#e-best-result)
    - [F. Other meshes](#f-other-meshes)
- [Experiments with Gaussian activation function](#experiments-with-gaussian-activation-function)
    - [100 epochs](#100-epochs)
    - [300 epochs](#300-epochs)
    - [500 epochs](#500-epochs)
- [Experiments with cross-entropy loss](#experiments-with-cross-entropy-loss)
  - [U mesh](#u-mesh-1)
    - [A. Changing number of anchors - 50 ceEpochs](#a-changing-number-of-anchors---50-ceepochs)
    - [B. Changing number of ceEpochs - 20 anchors](#b-changing-number-of-ceepochs---20-anchors)
  - [Dolphin mesh](#dolphin-mesh-1)
    - [A. Changing number of anchors - 50 ceEpochs](#a-changing-number-of-anchors---50-ceepochs-1)
    - [B. Changing number of ceEpochs - 100 anchors](#b-changing-number-of-ceepochs---100-anchors)
  - [Horse mesh](#horse-mesh-1)
    - [A. Changing number of anchors - 50 ceEpochs](#a-changing-number-of-anchors---50-ceepochs-2)
    - [B. Changing number of ceEpochs - 100 anchors](#b-changing-number-of-ceepochs---100-anchors-1)
  - [Dragon mesh](#dragon-mesh-1)
    - [A. Changing number of anchors - 50 ceEpochs](#a-changing-number-of-anchors---50-ceepochs-3)
    - [B. Changing number of ceEpochs - 200 anchors](#b-changing-number-of-ceepochs---200-anchors)
  - [Results using only the cross-entropy](#results-using-only-the-cross-entropy)
    - [Dolphin - 50 anchors](#dolphin---50-anchors)
    - [Dolphin - 100 anchors](#dolphin---100-anchors)
    - [Horse - 100 anchors](#horse---100-anchors)
    - [Dragon - 200 anchors](#dragon---200-anchors)
- [Experiments with cross-entropy loss and sparsemax activation](#experiments-with-cross-entropy-loss-and-sparsemax-activation)
  - [Dolphin mesh - 100 anchors](#dolphin-mesh---100-anchors)
    - [1. Sparsemax](#1-sparsemax)
    - [2. Entmax15](#2-entmax15)
  - [Dragon mesh - 200 anchors](#dragon-mesh---200-anchors)
    - [1. Sparsemax](#1-sparsemax-1)
    - [2. Entmax15](#2-entmax15-1)
  - [Comparison between training with cross-entropy, SDF or both](#comparison-between-training-with-cross-entropy-sdf-or-both)
    - [Dolphin - 50 anchors](#dolphin---50-anchors-1)
    - [Horse - 100 anchors](#horse---100-anchors-1)
    - [Dragon - 200 anchors](#dragon---200-anchors-1)


## Directory Structure

* **signedWeightTraining**: Scripts for training and testing anchor weights on 2D meshes using ReLU activation
* **signedWeightTrainingLoss**: Scripts for training and testing anchor weights on 2D meshes using Sigmoid activation and loss function
* **signedWeightTrainingSigma**: Scripts for training and testing anchor weights on 2D meshes using Sigmoid activation
* **sirenTraining**: Scripts for SIREN training and testing
* **Meshes**: Input mesh
* **training3D**: Scripts for training and testing on 3D meshes
* **training2D**: Scripts for training and testing on 2D meshes
* **weightTrainingReLu**: Scripts for training and testing anchor weights on 2D meshes using ReLU activation
* **weightTrainingSigma**: Scripts for training and testing anchor weights on 2D meshes using Sigmoid activation
* **weightTrainingSiren**: Scripts for training and testing anchor weights on 2D meshes using SIREN architecture

---

### Reconstructed 3D meshes (unsigned distance field with Lipschitz loss)
<img src="training3D/img/armadillo.png" width="400" alt="Reconstructed Armadillo 3D Mesh">

### Reconstructed 2D meshes (unsigned distance field with Lipschitz loss)
<p align="left">
  <img src="training2D/img/cheval.png" width="200">
  <img src="training2D/img/dauphin.png" width="200">
  <img src="training2D/img/U.png" width="200">
  <img src="training2D/img/Dragon.png" width="200">
</p>

---
## Training Unsigned Distance Fields

#### 1. Trained with Softmax activation on the final layer(level: 0.2)
[Link to code](weightTrainingRelu)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 35

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingRelu/img/neural_voronoi_diagram_labeled.png" width="400">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingRelu/img/anchor_influence_grid.png" width="400">


#### 1.1 Trained with SIREN architecture (level: 0.2)
[Link to code](weightTrainingSiren)
Best result with:
* input dimension: 2
* hidden dimension: 128
* number of layers: 6
* frequency multiplier ($\omega$~0~): 5
*	scaling factor (c): 6
* batch size: 256
* epochs: 500
* number of anchors: 100

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSiren/Img/voronoi_diagram_500ep.png" width="400" alt="SIREN Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSiren/Img/anchor_influence_grid_500ep.png" width="400">

#### 2. Trained with normalization formula on final layer and Signed distance fields as input
[Link to code](weightTrainingSigma)

##### Using ReLU (level: 0.05)

Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_U_Relu.png" width="400" alt="ReLU Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_U_Relu.png" width="400" alt="ReLU Influence Diagram">

##### Using Sigmoid (level: 0.2)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_U_sigma.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_U_Sigma.png" width="400" alt="Sigmoid Influence Diagram">

#### 3. Trained with normalization formula on final layer and Unsigned distance fields as input
[Link to code](weightTrainingSigma)

##### Using ReLU (level: 0.2)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 200
* number of anchors: 50

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_UDF_Relu.png" width="400" alt="ReLU Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_UDF_Relu.png" width="400" alt="ReLU Influence Diagram">

##### Using Sigmoid (level: 0.2)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50
* number of random points: 100000

###### 500 epochs and 10000 points
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_UDF500_sigma.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_UDF500.png" width="400" alt="Sigmoid Influence Diagram">

###### 100 epochs and 100000 points
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_UDF100_sigma.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_UDF100.png" width="400" alt="Sigmoid Influence Diagram">

---
## Training Signed Distance Fields

#### 1. Trained with Softmax activation on the final layer
[Link to code](signedWeightTraining)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="signedWeightTraining/img/neural_voronoi_diagram_U.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTraining/img/anchor_influence_grid.png" width="400" alt="Sigmoid Influence Diagram">

#### 2. Trained with normalization formula on final layer using Sigmoid
[Link to code](signedWeightTrainingSigma)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="signedWeightTrainingSigma/img/neural_voronoi_diagram_U.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingSigma/img/anchor_influence_grid.png" width="400" alt="Sigmoid Influence Diagram">

#### 3. Trained with normalization formula on final layer using Softmax and loss function
[Link to code](signedWeightTrainingLoss)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50


###### 50 anchors
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/neural_voronoi_diagram_labeled_50.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/anchor_influence_grid_50.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Anchor * Distance Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/weightDist_influence_grid_50.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Anchor Influence per Point</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/anchor_influence_count_50.png" width="400" alt="Sigmoid Influence Diagram">


###### 15 anchors
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/neural_voronoi_diagram_labeled_15.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/anchor_influence_grid_15.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Anchor * Distance Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/weightDist_influence_grid_15.png" width="400" alt="Sigmoid Influence Diagram">
<p align="center">Anchor Influence per Point</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/anchor_influence_count_15.png" width="400" alt="Sigmoid Influence Diagram">


---
## Experiments for better contours + loss function
Experiments done for signed fields with Softmax on final layer and Relu as activation function.

### U mesh

#### A. Changing number of anchors and epochs
Architecture: 
* batchSize = 128
* lm = 1.0
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000

##### 1. 15 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/neural_voronoi_diagram_labeled_15.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/15 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

##### 2. 50 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/neural_voronoi_diagram_labeled_50.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 3. 200 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/200 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/200 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

#### B. Changing the influence factor of the loss (lm)
Architecture: 
* batchSize = 128
* epochs = 100
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 50

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epochs 0.1 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/neural_voronoi_diagram_labeled_50.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1.0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch 1.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1.5 lm</b>
  </div>

</div>

#### C. Best result (50 anchors; 300 epochs) with change in lm
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epoch 0.9 lm - best/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1.0 lm</b>
  </div>

</div>


### Dolphin mesh

#### A. Changing number of anchors and epochs
Architecture: 
* batchSize = 128
* lm = 1.0
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000

##### 1. 50 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/50 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/50 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

##### 2. 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 3. 200 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/200 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/200 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

#### B. Changing the influence factor of the loss (lm)
Architecture: 
* batchSize = 128
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 100

##### 1. 100 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs 0.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs 0.5 lm 0.01 h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm and 0.01 h</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

##### 2. 300 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.1 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

##### 3. 500 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/500 epochs 0.5 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/500 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

#### C. Best result 


### Horse mesh

#### A. Changing number of anchors and epochs
Architecture: 
* batchSize = 128
* lm = 1.0
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000

##### 1. 300 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 2. 500 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/500 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

<div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/500 anchors/300 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

#### B. Changing the influence factor of the loss (lm)
Architecture: 
* batchSize = 128
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 300

##### 1. 300 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

### Dragon mesh

#### A. Changing number of anchors and epochs
Architecture: 
* batchSize = 128
* lm = 1.0
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000

##### 1. 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 2. 300 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 3. 500 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

#### B. Changing the influence factor of the loss (lm)
Architecture: 
* batchSize = 128
* epochs = 100
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 300

##### 1. 300 anchors
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 1.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1.5 lm</b>
  </div>

</div>

##### 2. 500 anchors
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

### Comparison

Different meshes trained at the same architecture but different number of anchors:

Architecture: 
* batchSize = 128
* epochs = 100
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 300
* lm = 0.9

U: 50 anchors
Dolphin: 100 anchors
Horse: 300 anchors
Dragon: 300 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epoch 0.9 lm - best/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
  </div>

</div>

### Experiments

Same mesh trained in different conditions:

Architecture: 
* batchSize = 128
* sigma = 0.01
* h = 0.2
* anchors = 100
* epochs = 100

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/100 a, 100 e, 256 hidden, 2 layers/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(1)</b>
  </div>
  <div>
    (1) architecture: 
    <ul>
      <li>hidden = 256</li>
      <li>noLayers = 2</li>
      <li>lm = 1.0</li>
      <li>noRandomPoints = 8000</li>
      <li>noSurfacePoints = 2000</li>
    </ul>
  </div> 

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/100 a, 100 e, 64 hidden, 6 layers/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(2)</b>
  </div>
  <div>
    (2) architecture: 
    <ul>
      <li>hidden = 64</li>
      <li>noLayers = 6</li>
      <li>lm = 1.0</li>
      <li>noRandomPoints = 8000</li>
      <li>noSurfacePoints = 2000</li>
    </ul>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/14K, 100 a, 100 e, 0.5 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(3)</b>
  </div>
  <div>
    (3) architecture: 
    <ul>
      <li>hidden = 64</li>
      <li>noLayers = 4</li>
      <li>lm = 0.5</li>
      <li>noRandomPoints = 10000</li>
      <li>noSurfacePoints = 4000</li>
    </ul>
  </div>


  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/14K, 100 a, 100 e, 1.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(4)</b>
  </div>
  <div>
    (4) architecture: 
    <ul>
      <li>hidden = 64</li>
      <li>noLayers = 4</li>
      <li>lm = 1.0</li>
      <li>noRandomPoints = 10000</li>
      <li>noSurfacePoints = 4000</li>
    </ul>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/100k, 100 a, 100 e, 1.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(5)</b>
  </div>
  <div>
    (5) architecture: 
    <ul>
      <li>hidden = 64</li>
      <li>noLayers = 4</li>
      <li>lm = 1.0</li>
      <li>noRandomPoints = 80000</li>
      <li>noSurfacePoints = 20000</li>
      <li>batchSize = 256</li>
    </ul>
  </div>
  
</div>

## Experiments with sigmoid activation function
Experiments done for signed fields with sigmoid function on final layer
[Link to code](signedWeightTrainingSigmaLoss)

Architecture:
* anchors = 100
* noRandomPoints = 2000
* noSurfacePoints = 8000
* inDim = 2
* hidden = 64
* noLayers = 4
* batchSize = 128
* epochs = 300
* lm = 1.0
* sigma = 0.01
* h = 0.2
  
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSigmaLoss/img/testing/100a 300e sigmoid/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Voronoi diagram and countours</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSigmaLoss/img/testing/100a 300e sigmoid/anchor_influence_grid.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Anchors influence</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSigmaLoss/img/testing/100a 300e sigmoid/weightDist_influence_grid.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>W*Dist</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSigmaLoss/img/testing/100a 300e sigmoid/anchor_influence_count.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Number of contributing anchors per point</b>
  </div>

</div>


## Experiments with sin activation function
Experiments done for signed fields with Softmax on final layer and sin function as activation layer (like SIREN architecture)
[Link to code](sineTrainingLoss)

#### A. Diferent architecture dimensions

Architecture: 
* sigma = 0.01
* h = 0.2
* lm = 1.0
* w0 = 30.0
* c = 6.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 100

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/64h 4l 128bs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(1)</b>
  </div>
  <div>
    (1) architecture: 
    <ul>
      <li>hidden = 64</li>
      <li>noLayers = 4</li>
      <li>batchSize = 128</li>
      <li>epochs = 100</li>
    </ul>
  </div> 

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/128h 6l 300e 250bs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(2)</b>
  </div>
  <div>
    (2) architecture: 
    <ul>
      <li>hidden = 128</li>
      <li>noLayers = 6</li>
      <li>batchSize = 250</li>
      <li>epochs = 300</li>
    </ul>
  </div>

 <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 100a 1000e 0.1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(3)</b>
  </div>
  <div>
    (3) architecture: 
    <ul>
      <li>hidden = 128</li>
      <li>noLayers = 6</li>
      <li>batchSize = 250</li>
      <li>epochs = 1000</li>
    </ul>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/256w 8l 300e/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>(4)</b>
  </div>
  <div>
    (4) architecture: 
    <ul>
      <li>hidden = 256</li>
      <li>noLayers = 8</li>
      <li>batchSize = 250</li>
      <li>epochs = 300</li>
    </ul>
  </div>
  
</div>

#### B. Different values for w0

Architecture: 
* sigma = 0.01
* h = 0.2
* lm = 1.0
* c = 6.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 100
* hidden = 128
* noLayers = 6
* batchSize = 250
* epochs = 300

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>5 w0</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/15w/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>15 w0</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/128h 6l 300e 250bs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>30 w0</b>
  </div>

</div>


#### C. Different number of anchors
Architecture: 
* sigma = 0.01
* h = 0.2
* lm = 1.0
* w0 = 5.0
* c = 6.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* hidden = 128
* noLayers = 6
* batchSize = 250
* epochs = 300

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 3a 0.1lm 300e/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>3 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 10a 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>30 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 50a 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>50 anchors</b>
  </div>

</div>


#### D. Lower lambda factor
Architecture: 
* sigma = 0.01
* h = 0.2
* w0 = 5.0
* c = 6.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 30
* hidden = 128
* noLayers = 6
* batchSize = 250
* epochs = 300

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 0.1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30e 0.5lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

#### E. Best result
Architecture: 
* sigma = 0.01
* h = 0.2
* lm = 0.1
* w0 = 5.0
* c = 6.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 30
* hidden = 128
* noLayers = 6
* batchSize = 250
* epochs = 300

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 0.1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Voronoi diagram and countours</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 0.1lm/anchor_influence_grid.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Anchors influence</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 0.1lm/weightDist_influence_grid.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>W*Dist</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/5w 30a 0.1lm/anchor_influence_count.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Number of contributing anchors per point</b>
  </div>

</div>

#### F. Other meshes

Architecture: 
* sigma = 0.01
* h = 0.2
* lm = 0.1
* w0 = 5.0
* c = 6.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* anchors = 30
* hidden = 128
* noLayers = 6
* batchSize = 250
* epochs = 300

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/U/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>U</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/horse/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Horse</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="sineTrainingLoss/img/dragon/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Dragon</b>
  </div>

</div>


## Experiments with Gaussian activation function
Experiments done for signed fields with a Gaussian function on final layer
[Link to code](signedWeightTrainingLoss)

Architecture: 
* sigma = 0.01
* h = 0.2
* lm = 1.0
* noRandomPoints = 8000
* noSurfacePoints = 2000
* hidden = 64
* noLayers = 4
* batchSize = 128
* hGauss = 0.2

#### 100 epochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 100e 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 100e 0.5lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 100e 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 0 lm</b>
  </div>

</div>

#### 300 epochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/10a 300e 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/50a 300e 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 300e 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>

</div>

#### 500 epochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/50a 500e 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 500e 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>

</div>


## Experiments with cross-entropy loss
Experiments done initialization of the weights with cross-entropy loss.
[Link to code](signedWeightTrainingCELoss)

Architecture: 
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* hidden = 64
* noLayers = 4
* batchSize = 128
* epochs = 100
* lm = 0

### U mesh

#### A. Changing number of anchors - 50 ceEpochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/U/10a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/U/15a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>15 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/U/20a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 anchors</b>
  </div>

</div>

#### B. Changing number of ceEpochs - 20 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/U/20a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/U/20a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

</div>

### Dolphin mesh

#### A. Changing number of anchors - 50 ceEpochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/50a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 50ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 50ce 1lm 0.05h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 1 lm and 0.05 h</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 50ce 1lm 0.02h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 1 lm and 0.02 h</b>
  </div>

</div>

#### B. Changing number of ceEpochs - 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 10ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

</div>


### Horse mesh

#### A. Changing number of anchors - 50 ceEpochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/50a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/150a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>150 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 50ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 1 lm</b>
  </div>
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 50ce 1lm 0.02h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors and 1 lm and 0.02h</b>
  </div>

</div>

#### B. Changing number of ceEpochs - 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

</div>


### Dragon mesh

#### A. Changing number of anchors - 50 ceEpochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/150a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>150 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>200 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 1lm 0.02h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>200 anchors and 1 lm 0.02h
    </b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>200 anchors and 1 lm</b>
  </div>

</div>

#### B. Changing number of ceEpochs - 200 anchors
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

</div>


### Results using only the cross-entropy

#### Dolphin - 50 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/50a 100e 10ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/50a 100e 30ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>30 ceEpochs</b>
  </div>

</div>

#### Dolphin - 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/100a 100e 10ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/100a 100e 20ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/100a 100e 50ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>
  

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/100a 100e 100ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 ceEpochs</b>
  </div>

</div>

#### Horse - 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/horse/100a 100e 10ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/horse/100a 100e 20ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/horse/100a 100e 50ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

</div>

#### Dragon - 200 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 10ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>10 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 20ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 50ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 100ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>100 ceEpochs</b>
  </div>
</div>


## Experiments with cross-entropy loss and sparsemax activation
Experiments done with initialization of the weights with cross-entropy loss and sparsemax ans the final layer activation function.
[Link to code](signedWeightTrainingSparsemax)

Architecture: 
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* hidden = 64
* noLayers = 4
* batchSize = 128
* epochs = 100
* ce = 10

### Dolphin mesh - 100 anchors

#### 1. Sparsemax
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dolphin/sparsemax/100a 10ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dolphin/sparsemax/100a 10ce 0.5lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dolphin/sparsemax/100a 10ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dolphin/sparsemax/100a 10ce 1lm 0.02h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm and 0.02h</b>
  </div>

</div>

#### 2. Entmax15
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dolphin/entmax/100a 10ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dolphin/entmax/100a 10ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

</div>


### Dragon mesh - 200 anchors

#### 1. Sparsemax
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/sparsemax/200a 10ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/sparsemax/200a 10ce 0.5lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/sparsemax/200a 10ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/sparsemax/200a 10ce 1lm 0.02h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm and 0.02h</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/other sparsemax 200a 10ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Another Sparsemax implementation</b>
  </div>

</div>

#### 2. Entmax15
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/entmax/200a 10ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/entmax/200a 10ce 1lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSparsemax/img/dragon/entmax/200a 10ce 1lm 0.02h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm and 0.02h</b>
  </div>

</div>

### Comparison between training with cross-entropy, SDF or both

Architecture: 
* sigma = 0.01
* h = 0.2
* noRandomPoints = 8000
* noSurfacePoints = 2000
* hidden = 64
* noLayers = 4
* batchSize = 128
* lm = 0

#### Dolphin - 50 anchors

* epochs = 100
* ceEpochs = 10

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dolphin/50a 100e 10ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Cross-entropy</b>
  </div>


  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/50a 100e 10ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Both</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/50 anchors/100 epoch 0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>SDF</b>
  </div>

</div>

#### Horse - 100 anchors

* epochs = 100
* ceEpochs = 20

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/horse/100a 100e 20ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Cross-entropy</b>
  </div>


  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Both</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/100 anchors 100 epochs 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>SDF</b>
  </div>

</div>

#### Dragon - 200 anchors

* epochs = 100
* ceEpochs = 20

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 20ce/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Cross-entropy</b>
  </div>


  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>Both</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/200 anchors 100 epochs 0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 400px;" /><br/>
    <b>SDF</b>
  </div>

</div>