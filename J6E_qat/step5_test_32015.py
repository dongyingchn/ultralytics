      
import time
import os
j6e_port = "32015"
model_bin_path = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/J6E_qat/compile/yolo11/model.hbm"
model_name = os.path.basename(model_bin_path)
model_folder = os.path.dirname(model_bin_path)
j6e_folder = "/map/wzheng/"

def communicate_with_j6e():

    #############generate run.sh##################

    cmd_line = "scp -r -P "+j6e_port+" "+ model_bin_path +" "+ "root@remote.uj2.cc:"+j6e_folder
    print("processing ", cmd_line)
    os.system(cmd_line)

    cmd_line = "ssh -p " + j6e_port + " root@remote.uj2.cc \' cd "+j6e_folder+" && "
    cmd_line = cmd_line + "export LD_LIBRARY_PATH=/map/wzheng/lib:/app/lib:/app/pub/lib:/middleware/lib:/middleware/pub/lib:/usr/hobot/lib:/usr/hobot/lib/sensor:/system/lib:/system/usr/lib:/lib: &&"

    cmd_line = cmd_line + "./hrt_model_exec perf --model_file " + model_name + " --core_id=0 --frame_count=200  --perf_time=0  --thread_num=1  --profile_path=\".\" && "
    # cmd_line = cmd_line + "./hrt_model_exec perf --model_file " + model_name +  " --internal_use --core_id=0 --frame_count=200  --perf_time=0  --thread_num=1  --profile_path=\".\"\'"
    cmd_line = cmd_line + "./hrt_model_exec model_info --model_file " + model_name + " >>" + j6e_folder +"profiler.log\'"
    print("processing ", cmd_line)
    os.system(cmd_line)
    print("waiting 0.5 seconds...")
    time.sleep(0.5)
    print("copying results")
    os.system("scp -r -P " + j6e_port + " root@remote.uj2.cc:" + j6e_folder + "profiler.log " + model_folder)
    return



if __name__ == "__main__":
    print("finish")
    communicate_with_j6e()


    