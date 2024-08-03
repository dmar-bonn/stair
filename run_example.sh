#!/bin/bash
if [ $# -lt 3 ]; then
    echo "Usage: $0 scene_id run_num 'list_of_object_id'"
    exit 1
fi


scene=$1
run_num=$2
target_id=$3
experiment_path=${4:-"./experiment"}


##########################################################################


for RUN in $(seq 1 $run_num)
do 
    planner=uniform
    python plan.py --config $planner implicit  --exp_path $experiment_path/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 

    planner=coverage
    python plan.py --config $planner implicit  --exp_path $experiment_path/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 

    planner=max_distance
    python plan.py --config $planner implicit  --exp_path $experiment_path/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 

    planner=uncertainty_all
    python plan.py --config $planner implicit  --exp_path $experiment_path/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 

    planner=uncertainty_target
    python plan.py --config $planner implicit  --exp_path $experiment_path/$scene --exp_id $RUN --target_class_id $target_id
    python eval_nerf.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 
    python eval_mesh.py --test_path test_data/$scene --exp_path $experiment_path/$scene/$planner/$RUN 

    python plot.py --scene_path $experiment_path/$scene
done
