#!/bin/bash
#SBATCH --job-name=physnrpet
#SBATCH --partition=24g
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --output=/mnt/home/kenliu/physnrpet.out
#SBATCH --error=/mnt/home/kenliu/physnrpet.err

# Build container if it doesn't exist
if [ ! -f ~/physnrpet.sqsh ]; then
    echo "Building container..."
    enroot import -o /tmp/physnrpet.sqsh 'docker://pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel'
    enroot create --force --name physnrpet /tmp/physnrpet.sqsh
    enroot start --root --rw --mount /mnt:mnt physnrpet \
        pip install pytorch-lightning monai wandb natsort matplotlib Pillow nibabel
    enroot export --force --output ~/physnrpet.sqsh physnrpet
    enroot remove physnrpet
    echo "Container built."
fi

# Run Main.py inside container
enroot create --force --name physnrpet ~/physnrpet.sqsh
enroot start --root --rw --mount /mnt:mnt physnrpet \
    bash -c "cd /mnt/home/kenliu/PhysNRPET && python Main.py"
