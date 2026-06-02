# Neural Distance Fields


Run script using  `python -m <folder.script>` (no `.py` at the end)

## Directory Structure

* **Meshes**: Input mesh
* **training3D**: Scripts for training and testing on 3D meshes
* **training2D**: Scripts for training and testing on 2D meshes
* **sirenTraining**: Scripts for SIREN training and testing
* **weightTrainingReLu**: Scripts for training and testing anchor weights on 2D meshes using ReLU activation
* **weightTrainingSiren**: Scripts for training and testing anchor weights on 2D meshes using SIREN architecture
* **weightTrainingSigma**: Scripts for training and testing anchor weights on 2D meshes using Sigmoid activation

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
<img src="signedWeightTrainingLoss/img/neural_voronoi_diagram_labeled_50.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingLoss/img/anchor_influence_grid_50.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor * Distance Influence Grids</p>
<img src="signedWeightTrainingLoss/img/weightDist_influence_grid_50.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor Influence per Point</p>
<img src="signedWeightTrainingLoss/img/anchor_influence_count_50.png" width="500" alt="Sigmoid Influence Diagram">


###### 15 anchors
<p align="center">Reconstructed Mesh Contour and Neural Voronoi Regions</p>
<img src="signedWeightTrainingLoss/img/neural_voronoi_diagram_labeled_15.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Individual Anchor Influence Grids</p>
<img src="signedWeightTrainingLoss/img/anchor_influence_grid_15.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor * Distance Influence Grids</p>
<img src="signedWeightTrainingLoss/img/weightDist_influence_grid_15.png" width="900" alt="Sigmoid Influence Diagram">
<p align="center">Anchor Influence per Point</p>
<img src="signedWeightTrainingLoss/img/anchor_influence_count_15.png" width="500" alt="Sigmoid Influence Diagram">

