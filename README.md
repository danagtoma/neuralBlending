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
- [Experiments for better contours](#experiments-for-better-contours)
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
<img src="training3D/img/armadillo.png" width="300" alt="Reconstructed Armadillo 3D Mesh">

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
<img src="weightTrainingRelu/img/neural_voronoi_diagram_labeled.png" width="900">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingRelu/img/anchor_influence_grid.png" width="900">


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
<img src="weightTrainingSiren/Img/voronoi_diagram_500ep.png" width="900" alt="SIREN Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSiren/Img/anchor_influence_grid_500ep.png" width="900">

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
<img src="weightTrainingSigma/img/neural_voronoi_diagram_U_Relu.png" width="900" alt="ReLU Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_U_Relu.png" width="900" alt="ReLU Influence Diagram">

##### Using Sigmoid (level: 0.2)
Best result with:
* input dimension: 2
* hidden dimension: 64
* number of layers: 4
* batch size: 128
* epochs: 100
* number of anchors: 50

<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_U_sigma.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_U_Sigma.png" width="900" alt="Sigmoid Influence Diagram">

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
<img src="weightTrainingSigma/img/neural_voronoi_diagram_UDF_Relu.png" width="900" alt="ReLU Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_UDF_Relu.png" width="900" alt="ReLU Influence Diagram">

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
<img src="weightTrainingSigma/img/neural_voronoi_diagram_UDF500_sigma.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_UDF500.png" width="900" alt="Sigmoid Influence Diagram">

###### 100 epochs and 100000 points
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="weightTrainingSigma/img/neural_voronoi_diagram_UDF100_sigma.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="weightTrainingSigma/img/anchor_influence_grid_UDF100.png" width="900" alt="Sigmoid Influence Diagram">

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
<img src="signedWeightTraining/img/neural_voronoi_diagram_U.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTraining/img/anchor_influence_grid.png" width="900" alt="Sigmoid Influence Diagram">

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
<img src="signedWeightTrainingSigma/img/neural_voronoi_diagram_U.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingSigma/img/anchor_influence_grid.png" width="900" alt="Sigmoid Influence Diagram">

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
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/neural_voronoi_diagram_labeled_50.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/anchor_influence_grid_50.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor * Distance Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/weightDist_influence_grid_50.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor Influence per Point</p>
<img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/anchor_influence_count_50.png" width="500" alt="Sigmoid Influence Diagram">


###### 15 anchors
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/neural_voronoi_diagram_labeled_15.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/anchor_influence_grid_15.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor * Distance Influence Grids</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/weightDist_influence_grid_15.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor Influence per Point</p>
<img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/anchor_influence_count_15.png" width="500" alt="Sigmoid Influence Diagram">


---
## Experiments for better contours
Experiments done for signed fields with Softmax on final layer.

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
    <img src="signedWeightTrainingLoss/img/U/15 anchors/100 epochs/neural_voronoi_diagram_labeled_15.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/15 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

##### 2. 50 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/neural_voronoi_diagram_labeled_50.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 3. 200 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/200 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/200 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
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
* noRefPoints = 50

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epochs 0.1 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0.1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch/neural_voronoi_diagram_labeled_50.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>1.0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/100 epoch 1.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>1.5 lm</b>
  </div>

</div>

#### C. Best result (50 anchors; 300 epochs) with change in lm
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epoch 0.9 lm - best/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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
    <img src="signedWeightTrainingLoss/img/dolphin/50 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/50 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

</div>

##### 2. 100 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 3. 200 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/200 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/200 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
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
* noRefPoints = 100

##### 1. 100 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs 0.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs 0.5 lm 0.01 h/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm and 0.01 h</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

##### 2. 300 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.1 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0.1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>1 lm</b>
  </div>

</div>

##### 3. 500 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/500 epochs 0.5 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/500 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 2. 500 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/500 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

<div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/500 anchors/300 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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
* noRefPoints = 300

##### 1. 300 epochs
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
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
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 2. 300 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>500 epochs</b>
  </div>

</div>

##### 3. 500 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/100 epochs/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>100 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>300 epochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/500 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
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
* noRefPoints = 300

##### 1. 300 anchors
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 1.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>1.5 lm</b>
  </div>

</div>

##### 2. 500 anchors
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs 0.5 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
    <b>0.9 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/300 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
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
* noRefPoints = 300
* lm = 0.9

U: 50 anchors
Dolphin: 100 anchors
Horse: 300 anchors
Dragon: 300 anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/U/50 anchors/300 epoch 0.9 lm - best/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/100 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/horse/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0.9 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 600px;" /><br/>
  </div>

</div>

### Experiments

Same mesh trained in different conditions:

Architecture: 
* batchSize = 128
* sigma = 0.01
* h = 0.2
* noRefPoints = 100
* epochs = 100

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/100 a, 100 e, 256 hidden, 2 layers/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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
    <img src="signedWeightTrainingLoss/img/tests/100 a, 100 e, 64 hidden, 6 layers/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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
    <img src="signedWeightTrainingLoss/img/tests/14K, 100 a, 100 e, 0.5 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>(3)</b>
  </div>
  <div>
    (3) architecture: 
    <ul>
      <li>hidden = 65</li>
      <li>noLayers = 4</li>
      <li>lm = 0.5</li>
      <li>noRandomPoints = 10000</li>
      <li>noSurfacePoints = 4000</li>
    </ul>
  </div>


  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/tests/14K, 100 a, 100 e, 1.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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
    <img src="signedWeightTrainingLoss/img/tests/100k, 100 a, 100 e, 1.0 lm/neural_voronoi_diagram_labeled.png"  style="width: 100%; max-width: 600px;" /><br/>
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

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingSigmaLoss/img/testing/100a, 300 e/neural_voronoi_diagram_labeled.png.png"  style="width: 100%; max-width: 600px;" /><br/>
    <b>(6) Modified activation function</b>
  </div>
  <div>
    (6) architecture: 
    <ul>
      <li>hidden = 64</li>
      <li>noLayers = 4</li>
      <li>lm = 1.0</li>
      <li>noRandomPoints = 8000</li>
      <li>noSurfacePoints = 2000</li>
      <li>epochs = 300</li>
      <li>noRefPoints = 100</li>
    </ul>
  </div>
  
</div>
