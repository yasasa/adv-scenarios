from scripts.datastore import Datastore, DatastoreDataset
from cubeadv.sim.sensors.sensor_utils import get_ndc_ray_grid

def fix_depth(depth):
    dirs = get_ndc_ray_grid(200, 66, 1.)
    dirs_im = dirs.view(66, 200, 3)
    print(depth.shape)
    depth = depth / dirs_im[:, :, 0].abs()
    return depth.numpy()
    
    
def apply_data_op(op, input_db_path, output_db_path, data_index):
    input = Datastore(-1, path=input_db_path)
    output = Datastore(input.size, path=output_db_path)
    
    for i in range(input.size):
        print(i)
        data = input[i]
        proc_data = op(data[data_index+1])
        output[i] = (data[0], proc_data)
    
    output.sync()

if __name__ == '__main__':
    op = fix_depth
    data_index = 3
    apply_data_op(op, "../experiments/depth-policy-2/testing_db", "../experiments/depth-policy-training-data/testing_db", data_index)
    apply_data_op(op, "../experiments/depth-policy-2/training_db", "../experiments/depth-policy-training-data/training_db", data_index)