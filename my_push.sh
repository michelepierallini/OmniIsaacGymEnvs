#!/bin/bash
git add omniisaacgymenvs/tasks/fishing_rod_real_pos_2.py
git add omniisaacgymenvs/tasks/fishing_rod_real_pos_2_cv.py
git add omniisaacgymenvs/cfg/task/FishingRodPosDueCV.yaml
git add omniisaacgymenvs/cfg/train/FishingRodPosDueCVPPO.yaml

git add omniisaacgymenvs/pyenvs.sh
git add omniisaacgymenvs/tasks/fishing_rod_real_pos_2_cv_model_based.py
git add omniisaacgymenvs/cfg/task/FishingRodPosDueCVMB.yaml
git add omniisaacgymenvs/cfg/train/FishingRodPosDueCVMBPPO.yaml
git add omniisaacgymenvs/tasks/fishing_rod_lumped_param/
git add omniisaacgymenvs/utils/task_util.py

git add omniisaacgymenvs/tasks/fishing_rod_real_pos_2_cv_model_based_pos.py
git add omniisaacgymenvs/cfg/train/FishingRodPosDueCVMBPosPPO.yaml
git add omniisaacgymenvs/cfg/task/FishingRodPosDueCVMBPos.yaml

git add omniisaacgymenvs/tasks/plot_from_tensorboard.py

git add my_push.sh 
git add .gitignore

# #######################################################
# if [ -n "$1" ]; then
#   COMMIT_MESSAGE="$1"
# else
#   COMMIT_MESSAGE="minor_changes"
# fi
# git commit -m "$COMMIT_MESSAGE"
# #######################################################
git commit -m "Update .gitignore and add fishing rod tasks plus model-based [working]"

# Push to the fork
git push my-fork main


