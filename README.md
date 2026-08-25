# Neural Distance Blending for Implicit Geometry Representation <!-- omit from toc -->

## Table of Contents <!-- omit from toc -->
- [Directory Structure](#directory-structure)
- [Experiments Using Locality Regularisation](#experiments-using-locality-regularisation)
  - [A. Changing the Number of Anchors and Epochs](#a-changing-the-number-of-anchors-and-epochs)
    - [1. 100 Anchors](#1-100-anchors)
    - [2. 500 Anchors](#2-500-anchors)
  - [B. Changing the Influence Factor of the Loss (lm)](#b-changing-the-influence-factor-of-the-loss-lm)
    - [1. 300 Anchors](#1-300-anchors)
    - [2. 500 Anchors](#2-500-anchors-1)
- [Experiments with Cross-Entropy Loss](#experiments-with-cross-entropy-loss)
  - [A. Changing the Number of Anchors (50 ceEpochs)](#a-changing-the-number-of-anchors-50-ceepochs)
  - [B. Changing the Number of ceEpochs (200 Anchors)](#b-changing-the-number-of-ceepochs-200-anchors)
- [Results Using Only Cross-Entropy Loss](#results-using-only-cross-entropy-loss)
  - [200 Anchors](#200-anchors)
- [Experiments with Gaussian Activation Function](#experiments-with-gaussian-activation-function)
  - [100 Epochs](#100-epochs)
  - [500 Epochs](#500-epochs)
- [Experiments with Sparsemax and Entmax Activations](#experiments-with-sparsemax-and-entmax-activations)
  - [1. Sparsemax](#1-sparsemax)
  - [2. Entmax](#2-entmax)
- [Comparison: Cross-Entropy, SDF, and Combined Training](#comparison-cross-entropy-sdf-and-combined-training)
- [SIREN Method for 2D Meshes](#siren-method-for-2d-meshes)
- [Ideal Voronoi Diagram and Exponential Weights](#ideal-voronoi-diagram-and-exponential-weights)
- [Results for 2D Meshes](#results-for-2d-meshes)
- [Results for 3D Meshes](#results-for-3d-meshes)
- [Lucy (High-Resolution Reconstruction)](#lucy-high-resolution-reconstruction)

---

## Directory Structure

* **Meshes**: Input mesh files.
* **signedWeightTrainingCELoss**: Scripts for training and testing anchor weights on 2D meshes using cross-entropy pretraining.
* **signedWeightTrainingCELoss3D**: Scripts for training and testing anchor weights on 3D meshes using cross-entropy pretraining.
* **signedWeightTrainingCELoss3DSiren**: Scripts for training and testing SIREN models on 3D meshes.
* **signedWeightTrainingCELossSiren**: Scripts for training and testing SIREN models on 2D meshes.
* **signedWeightTrainingLoss**: Scripts for training and testing anchor weights on 2D meshes with different architectures and loss functions.
* **otherTests**: Miscellaneous and unorganised test scripts.

---

## Experiments Using Locality Regularisation
Experiments conducted using a locality regularisation loss during SDF training.  
[Link to code](signedWeightTrainingLoss)

### A. Changing the Number of Anchors and Epochs
**Hyperparameters:**
* `batchSize` = 128
* `lm` = 1.0
* `sigma` = 0.01
* `h` = 0.2
* `noQueryPoints` = 10000

#### 1. 100 Anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/100 anchors/100 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
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

#### 2. 500 Anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/500 anchors/100 epochs/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
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

---

### B. Changing the Influence Factor of the Loss (lm)
**Hyperparameters:**
* `batchSize` = 128
* `epochs` = 100
* `sigma` = 0.01
* `h` = 0.2
* `noQueryPoints` = 10000

#### 1. 300 Anchors
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/300 anchors/300 epochs 0 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
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

#### 2. 500 Anchors
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

---

## Experiments with Cross-Entropy Loss
Experiments conducted with weight pretraining via cross-entropy loss.  
[Link to code](signedWeightTrainingCELoss)

**Hyperparameters:**
* `sigma` = 0.01
* `h` = 0.2
* `noQueryPoints` = 10000
* `hidden` = 64
* `noLayers` = 4
* `batchSize` = 128
* `epochs` = 100

### A. Changing the Number of Anchors (50 ceEpochs)

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors, 0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/150a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>150 anchors, 0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>200 anchors, 0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>200 anchors, 1 lm</b>
  </div>
</div>

### B. Changing the Number of ceEpochs (200 Anchors)
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>
</div>

---

## Results Using Only Cross-Entropy Loss
Experiments using exclusively cross-entropy pretraining for the weights.

### 200 Anchors

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 10ce/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>10 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 20ce/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>20 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 50ce/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>50 ceEpochs</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 100ce/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>100 ceEpochs</b>
  </div>
</div>

---

## Experiments with Gaussian Activation Function
Experiments using a Gaussian activation function on the final layer.  
[Link to code](signedWeightTrainingLoss)

**Hyperparameters:**
* `sigma` = 0.01
* `h` = 0.2
* `lm` = 1.0
* `noQueryPoints` = 10000
* `hidden` = 64
* `noLayers` = 4
* `batchSize` = 128
* `hGauss` = 0.2

### 100 Epochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 100e 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors, 1 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 100e 0.5lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors, 0.5 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 100e 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors, 0 lm</b>
  </div>
</div>

### 500 Epochs

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/50a 500e 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>50 anchors</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dolphin/Gaussian/100a 500e 1lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>100 anchors</b>
  </div>
</div>

---

## Experiments with Sparsemax and Entmax Activations
Experiments evaluating Sparsemax and Entmax as final-layer activation functions.  
[Link to code](signedWeightTrainingLoss)

**Hyperparameters:**
* `sigma` = 0.01
* `h` = 0.2
* `noQueryPoints` = 10000
* `hidden` = 64
* `noLayers` = 4
* `batchSize` = 128
* `epochs` = 100
* `anchors` = 200

### 1. Sparsemax
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/report/sparsemax 0lm/sparse0lm_neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/report/sparsemax 1lm/sparse1lm_neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>
</div>

### 2. Entmax
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/report/entmax 0lm/ent0lm_neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>0 lm</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/report/entmax 1lm/ent1lm_neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>1 lm</b>
  </div>
</div>

---

## Comparison: Cross-Entropy, SDF, and Combined Training

**Hyperparameters:**
* `sigma` = 0.01
* `h` = 0.2
* `noQueryPoints` = 10000
* `hidden` = 64
* `noLayers` = 4
* `batchSize` = 128
* `lm` = 0
* `epochs` = 100
* `ceEpochs` = 20
* `anchors` = 200

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/only CE/dragon/200a 100e 20ce/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Cross-Entropy Only</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 20ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Combined (CE + SDF)</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingLoss/img/dragon/200 anchors 100 epochs 0 lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>SDF Only</b>
  </div>
</div>

---

## SIREN Method for 2D Meshes
Experiments conducted using the SIREN implicit representation on 2D meshes.  
[Link to code](signedWeightTrainingCELossSiren)

**Hyperparameters:**
* `noQueryPoints` = 10000
* `hidden` = 64
* `noLayers` = 6
* `batchSize` = 128
* `epochs` = 100
* `w0` = 30
* `c` = 6

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELossSiren/img/dragon/sirenMethod_contours_dragon.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>SIREN</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 0lm/levelSet_contoursdragon.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Ours</b>
  </div>
</div>

---

## Ideal Voronoi Diagram and Exponential Weights

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
     <img src="signedWeightTrainingCELoss/voronoi_and_sdf_contours.png" style="width: 100%; max-width: 300px;" /><br/>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200_ideal_exponential_weights.png" style="width: 100%; max-width: 300px;" /><br/>
  </div>
</div>

---

## Results for 2D Meshes
**Hyperparameters:**
* `noQueryPoints` = 10000
* `hidden` = 64
* `noLayers` = 4
* `batchSize` = 128
* `epochs` = 100
* `ceEpochs` = 50

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 30px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/U/20a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>U-Shape (20 anchors)</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dauphin/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Dolphin (100 anchors)</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/horse/100a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Horse (100 anchors)</b>
  </div>

  <div style="text-align: center; margin-bottom: 30px;">
    <img src="signedWeightTrainingCELoss/img/dragon/200a 100e 50ce 0lm/neural_voronoi_diagram_labeled.png" style="width: 100%; max-width: 400px;" /><br/>
    <b>Dragon (200 anchors)</b>
  </div>
</div>

---

## Results for 3D Meshes
Trained on 15,000 anchors and 100,000 random query points.

**Hyperparameters:**
* `hidden` = 128
* `noLayers` = 6
* `batchSize` = 256
* `epochs` = 1000
* `ceEpochs` = 100

<p align="center">
  <img src="abacaOutput/15k 100k 100ce 1000e/armadillo/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/bunny/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/cheburashka/screenshot_000000.png" width="400"> 
  <img src="abacaOutput/15k 100k 100ce 1000e/dragon/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/Dragon_2/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/happy/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/lucy/recon.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/max-planck/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/Ram/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/Scallop/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/Glykon/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/cellular_lamp/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/LightBulb/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/MedievalCastle/screenshot_000000.png" width="400">
  <img src="abacaOutput/15k 100k 100ce 1000e/Buckyballs/screenshot_000000.png" width="400">
</p>

---

## Lucy (High-Resolution Reconstruction)
Trained on 100,000 anchors and 1,000,000 random query points.

**Hyperparameters:**
* `hidden` = 128
* `noLayers` = 6
* `batchSize` = 256
* `epochs` = 500
* `ceEpochs` = 100

<p align="center">
  <img src="abacaOutput/lucy 100k 1mil/reconstructedMesh_bigLucy.png" width="1000" alt="High resolution Lucy reconstruction">
</p>