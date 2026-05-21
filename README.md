<!-- # Neural Distance Fields

run script with  `python -m <folder.script>` (no py at the end)

## Directory Structure

<ul>
    <li>Meshes: Input meshes </li>
    <li>training3D: Scripts for training and testing on 3D meshes</li>
    <li>training2D: Scripts for training and testing on 2D meshes</li>
    <li>sirenTraining: Scripts for SIREN training and testing.</li>
    <li>weightTrainingReLu: Scripts for training and testing anchors` weights for 2D meshes using ReLu activation</li>
    <li>weightTrainingSiren: Scripts for training and testing anchors` weights for 2D meshes using SIREN architecture</li>
    <li>weightTrainingSigma: Scripts for training and testing anchors` weights for 2D meshes using Sigmoid activation</li>
</ul>


## Some results

##### Reconstructed mesh using unsigned distance fields for 3D meshes with Lipschitz loss

<img src="training3D/img/armadillo.png" width="300">


##### Reconstructed mesh using unsigned distance fields at for 2D meshes with Lipschitz loss

<img src="training2D/img/cheval.png" width="300">
<img src="training2D/img/dauphin.png" width="300">
<img src="training2D/img/U.png" width="300">
<img src="training2D/img/Dragon.png" width="300">


##### Influence diagram and contours at level 0.2 for anchors' weights trained with softmax activation on last layer

<img src="weightTrainingRelu/img/neural_voronoi_diagram_labeled.png" width="900">


##### Influence diagram and contours at level 0.2 for anchors' weights trained with Siren architecture

<img src="weightTrainingSiren/Img/voronoi_diagram_500ep.png" width="900">


##### Influence diagram and contours at level 0.2 for anchors' weights trained with ..... on last layer

Using ReLu as the function
<img src="weightTrainingSigma/img/neural_voronoi_diagram_labeled_U_Relu.png" width="900">

Using Sigmoid as the function
<img src="weightTrainingSigma/img/neural_voronoi_diagram_labeled_U_sigma.png" width="900"> -->

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

## Results

### Reconstructed 3D meshes (unsigned distance field with Lipschitz loss)
<img src="training3D/img/armadillo.png" width="300" alt="Reconstructed Armadillo 3D Mesh">

### Reconstructed 2D meshes (unsigned distance field with Lipschitz loss)
<p align="left">
  <img src="training2D/img/cheval.png" width="200">
  <img src="training2D/img/dauphin.png" width="200">
  <img src="training2D/img/U.png" width="200">
  <img src="training2D/img/Dragon.png" width="200">
</p>

### Influence diagrams and Contours

#### Trained with Softmax activation on the final layer (level: 0.2)
<img src="weightTrainingRelu/img/neural_voronoi_diagram_labeled.png" width="900" alt="Softmax Influence Diagram">

#### Trained with SIREN architecture (level: 0.2)
<img src="weightTrainingSiren/Img/voronoi_diagram_500ep.png" width="900" alt="SIREN Influence Diagram">

#### Trained with normalization formula on final layer

##### Using ReLU (level: 0.05)
<img src="weightTrainingSigma/img/neural_voronoi_diagram_labeled_U_Relu.png" width="900" alt="ReLU Influence Diagram">

##### Using Sigmoid (level: 0.2)
<img src="weightTrainingSigma/img/neural_voronoi_diagram_labeled_U_sigma.png" width="900" alt="Sigmoid Influence Diagram">