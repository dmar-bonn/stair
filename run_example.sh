#!/bin/bash
experiment_name=experiment1
scene=scene2
target_id="6"

##########################################################################


for RUN in 1 2 3 4 5
do 
    planner=uniform
    python plan.py --config $planner implicit  --exp_name $experiment_name/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 

    planner=coverage
    python plan.py --config $planner implicit  --exp_name $experiment_name/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 

    planner=max_distance
    python plan.py --config $planner implicit  --exp_name $experiment_name/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 

    planner=uncertainty_all
    python plan.py --config $planner implicit  --exp_name $experiment_name/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 

    planner=uncertainty_target
    python plan.py --config $planner implicit  --exp_name $experiment_name/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path experiment/$experiment_name/$scene/$planner/$RUN 

    python plot.py --scene_path experiment/$experiment_name/$scene
done
