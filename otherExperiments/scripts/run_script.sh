#!/bin/bash
#OAR -q abaca
#OAR -l gpu=1,walltime=00:30:00
#OAR -O outputs/OAR_%jobid%.out
#OAR -E outputs/OAR_%jobid%.err

module load cuda/12.1.1_gcc-10.4.0

source pyEnv/bin/activate

set -x

MESHES=("Buckyballs" "lucy" "bunny" "cheburashka" "Dragon_2" "dragon" "Glykon" "happy" "LightBulb" "max-planck" "MedievalCastle" "Ram" "Scallop")

for MESH in "${MESHES[@]}"
do
    mkdir -p "outputs/${MESH}"
    
    {
        echo "Step 1: Sampling Points"
        python scripts/samplingPoints3D.py "$MESH"
        
        echo "Step 2: Training Model"
        python -u scripts/training.py "$MESH"

	echo "Step 3: Running Tests"
        python scripts/testing.py "$MESH"

	echo "Gradient Norm"
        python scripts/gradientNorm.py "$MESH"

	echo "Anchor Influence"
	python scripts/anchorInfluence.py "$MESH"
    } > "outputs/${MESH}/console_output.log" 2>&1
done

echo "All cluster pipeline tasks completed successfully."
