#!/bin/bash
# content_path="/home/yangxl21/DVI_data/ViT_CIFAR10"
# END=15
# # N_jobs=10

# for iteration in {11..15}; do
#     python ${script} --content_path ${content_path} --setting tdvi -i ${iteration} -v tdvi 1> ${content_path}/log_vis/${iteration}.log 2>&1 &
#     sleep 35s
#     # if [[ $(jobs -r -p | wc -l) -ge $N_jobs ]]; then
#     #     wait -n
#     # fi
# done
# # wait
# echo "All Done!"

# content_path="/home/yangxl21/DVI_data/cl"
# resumes=(0 1 2 3 4)
# iterations=(1 2 3 4 5)

# for i in {0..4}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done

# content_path="/home/yangxl21/DVI_data/class_ewc"
# resumes=(0 1 2 3 4)
# iterations=(1 2 3 4 5)

# for i in {0..4}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done


content_path="/home/yangxl21/DVI_data/class_fromp"
resumes=(0 1 2 3 4)
iterations=(1 2 3 4 5)

for i in {0..4}; 
do
    nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
done


content_path="/home/yangxl21/DVI_data/class_replay"
resumes=(0 1 2 3 4)
iterations=(1 2 3 4 5)

for i in {0..4}; 
do
    nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
done


# content_path="/home/yangxl21/DVI_data/domain_ewc"
# resumes=(0 1 2 3 4)
# iterations=(1 2 3 4 5)

# for i in {0..4}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done

content_path="/home/yangxl21/DVI_data/domain_replay"
resumes=(0 1 2 3 4)
iterations=(1 2 3 4 5)

for i in {0..4}; 
do
    nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
done
content_path="/home/yangxl21/DVI_data/domain_fromp"
resumes=(0 1 2 3 4)
iterations=(1 2 3 4 5)

for i in {0..4}; 
do
    nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
done