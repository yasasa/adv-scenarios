
## train with PID data
#python train_VPF.py --iter 0
#./automate.sh

# prerequisite:
# 1. run PID controller to collect data to dagger_data/ep_0
# 2. open Carla server with config: python config.py -m Town01 --fps 20

# run PID controller to collect data to expert/ep_0
mkdir ./random_sampling/$(date '+%Y-%m-%d.')$1
# mkdir ./random_sampling/$(date '+%Y-%m-%d.')$1/no_cube
# mkdir ./random_sampling/$(date '+%Y-%m-%d.')$1/with_cube
# # open simulator

echo "open simulator --------------------------------------------------------------------"
/home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-19-g318a7ff5-dirty/LinuxNoEditor/CarlaUE4.sh &
sleep 15
echo "setting FPS to 20 -----------------------------------------------------------------"
python /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-19-g318a7ff5-dirty/LinuxNoEditor/PythonAPI/util/config.py -m Town01 --fps 20
sleep 15
# echo "collect expert data -----------------------------------------------------------------"
# python new_PID_collect.py --expnum $1 | tee ./experiments/$(date '+%Y-%m-%d.')$1/PID_collect.out

# pkill CarlaUE4
# sleep 5
# pkill CarlaUE4
# sleep 5

# initial training
for (( i=0; i<=20; i++ ))
do
    echo "no cube iteration $i --------------------------------------------------------------------"
    python random_sample_policy.py --iter $i --expnum $1 --policy 4 | tee ./random_sampling/$(date '+%Y-%m-%d.')$1/ep_$i/no_cube.out &
    pid=$!
    wait $pid

    sleep 15

    echo "with cube iteration $i"
    python random_sample_policy.py --iter $i --expnum $1 --cube True --policy 4 | tee ./random_sampling/$(date '+%Y-%m-%d.')$1/ep_$i/with_cube.out &
    pid=$!
    python spawn_npc.py --expnum $1 --iter $i &
    pid2=$!
    wait $pid
    wait $pid2

    sleep 15
done