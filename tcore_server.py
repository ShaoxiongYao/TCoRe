import os
import time
import numpy as np

if __name__ == "__main__":
    while True:
        while os.path.exists("ready_client.txt") == False:
            time.sleep(0.1)
        time.sleep(1)
        strs = np.loadtxt("client_strs.txt", dtype=str)
        os.remove("ready_client.txt")
        cmd = "python tcore_infer_once.py"
        cmd += " --w checkpoints/pretrained_model.ckpt"
        cmd += " --input_file_name " + strs[0]
        cmd += " --output_file_name " + strs[1]
        os.system(cmd)
        f = open("ready_server.txt", "a")
        f.close()
