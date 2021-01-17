
## train with PID data
#python train_VPF.py --iter 0
#./automate.sh

# prerequisite:
# 1. run PID controller to collect data to dagger_data/ep_0
# 2. open Carla server with config: python config.py -m Town01 --fps 20

# run PID controller to collect data to expert/ep_0
# mkdir ./experiments/$(date '+%Y-%m-%d.')$1
mkdir ./experiments/$(date '+%Y-%m-%d.')$1/model
mkdir ./experiments/$(date '+%Y-%m-%d.')$1/data
# # open simulator

# echo "open simulator --------------------------------------------------------------------"
# /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-19-g318a7ff5-dirty/LinuxNoEditor/CarlaUE4.sh &
# sleep 15
# echo "setting FPS to 20 -----------------------------------------------------------------"
# python /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-19-g318a7ff5-dirty/LinuxNoEditor/PythonAPI/util/config.py -m Town01 --fps 20
# sleep 15
# echo "collect expert data -----------------------------------------------------------------"
# python new_PID_collect.py --expnum $1 | tee ./experiments/$(date '+%Y-%m-%d.')$1/PID_collect.out

# pkill CarlaUE4
# sleep 5
# pkill CarlaUE4
# sleep 5

# initial training
for (( i=0; i<=5; i++ ))
do
    echo "training iteration $i --------------------------------------------------------------------"
    python train_CNN.py --iter $i --expnum $1 | tee ./experiments/$(date '+%Y-%m-%d.')$1/data/ep_$i/train_CNN.out

    sleep 5
    #echo "Welcome $i times"
    # open simulator
    echo "open simulator --------------------------------------------------------------------"
    /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-19-g318a7ff5-dirty/LinuxNoEditor/CarlaUE4.sh &
    sleep 20

    echo "setting FPS to 20 -----------------------------------------------------------------"
    python /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-19-g318a7ff5-dirty/LinuxNoEditor/PythonAPI/util/config.py -m Town01 --fps 20
    sleep 20

    echo "recollect data --------------------------------------------------------------------"
    python collect_CNN.py --iter $i --expnum $1 | tee ./experiments/$(date '+%Y-%m-%d.')$1/data/ep_$i/collect_CNN.out

    pkill CarlaUE4
    sleep 5
    pkill CarlaUE4
    sleep 5

    code=$?
    echo $code
    if (($code > 0)); then
        echo "ext fail - $code : stop"
        # kill simulator
        pkill CarlaUE4
        sleep 5
        pkill CarlaUE4
        break
    fi
    wait
    sleep 2
done






# ## train with PID data
# #python train_VPF.py --iter 0
# #./automate.sh


# # initial training
# echo "init training --------------------------------------------------------------------"
# python train_VPF.py --iter 0 | tee ./dagger_data/ep_0/train_out.log

# wait
# sleep 2

# for (( i=1; i<=5; i++ ))
# do
#     #echo "Welcome $i times"
#     # open simulator
#     echo "open simulator --------------------------------------------------------------------"
#     /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-5-g255683a7-dirty/LinuxNoEditor/CarlaUE4.sh &
#     #/home/zidong/Desktop/CARLA_0.9.5/CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=20 > ./dagger_data/ep_$(expr $i - 1)/server_out.log &
#     #pid=$!

#     sleep 7

#     echo "setting FPS to 20 -----------------------------------------------------------------"
#     python /home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-5-g255683a7-dirty/LinuxNoEditor/PythonAPI/util/config.py -m Town01 --fps 20

#     #echo $pid
#     # recollect
#     echo "recollect data --------------------------------------------------------------------"
#     python NN_collect.py --test 0 --iter $(expr $i - 1) | tee ./dagger_data/ep_$(expr $i - 1)/collect_out.log
#     #python NNPID_collect.py --test 0 --fix 0 --iter $(expr $i - 1)    # | tee ./dagger_data/ep_$(expr $i - 1)/collect_out.log
    
#     code=$?
#     echo $code
#     if (($code == 100)); then
#         echo "ext fail - 100 : frame skiped"
#         # kill simulator
#         pkill CarlaUE4
#         sleep 5
#         pkill CarlaUE4
#         break
#     elif (($code > 0)); then
#         echo "ext fail - $code : stop"
#         # kill simulator
#         pkill CarlaUE4
#         sleep 5
#         pkill CarlaUE4
#         break
#     fi
    
#     sleep 2

#     # kill simulator
#     pkill CarlaUE4
#     sleep 5
#     pkill CarlaUE4
    
#     echo "training iteration $i --------------------------------------------------------------------"
#     python train_VPF.py --iter $i | tee ./dagger_data/ep_$i/train_out.log
    
#     code=$?
#     echo $code
#     if (($code > 0)); then
#         echo "ext fail - $code : stop"
#         # kill simulator
#         pkill CarlaUE4
#         sleep 5
#         pkill CarlaUE4
#         break
#     fi
    
#     wait
#     sleep 2
# done



