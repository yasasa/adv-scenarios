
def generate_cnn_lidar_data (path, episode, batch_count, batch_size):
    # episodes
    for ep in {episode}:
        for i in range(batch_count):
            all_batch_data = []
            ## create directory if not exists
            try:
                os.stat(path + str(ep) + '/Spherical')
            except:
                os.mkdir(path + str(ep) + '/Spherical')
            ## test if the file exist
            filePath = path + str(ep) + '/Lidar/' + str(i) + '.npy'
            print(filePath)
            if not os.path.exists(filePath):
                print(filePath,'doesnt exist')
                print('its ok, skip')
                continue
            
            print('loading.. '+ filePath)
            lidar = np.load(filePath)

            lidar = lidar.reshape((lidar.shape[0], lidar.shape[1], 3))

            for index in range(batch_size):
                ## for Lidar data
                points = lidar[index]
                depth1d = combine_cnn_lidar_data(points)

                all_batch_data.append(depth1d)
                # print("all_batch_data length is: ", len(all_batch_data))

            all_batch_data = np.asarray(all_batch_data)
            # print("all_batch_data shape is: ", all_batch_data.shape)
            # plt.show()
            save_name = path + str(ep) + '/Spherical/' + str(i) + '.npy'
            # print("combined lidar rgb saved in file: ", save_name)
            np.save(save_name, all_batch_data)
            # print(np.load(save_name))
    print("generate_cnn_lidar_data is done")



# num_batch per episode
def load_data_colour_lidar (path, episode_start, episode_end, batch_num_start, batch_num_end):
    ## Training Data
    data_img = []
    data_ctrl = []
    data_pts = []
    
    # episodes
    for ep in range(episode_start, episode_end, 1):
        # images - 100 per batch
        for i in range(batch_num_start, batch_num_end, 1):   # 2 to 43
            ## test if the file exist
            filePath = path + str(ep) + '/Coloured_Lidar/' + str(i) + '.npy'
            if not os.path.exists(filePath):
                print(filePath,'doesnt exist')
                print('its ok, skip')
                continue
            
            pts = np.load(path + str(ep) + '/Coloured_Lidar/' + str(i) + '.npy')
            ctrls = np.load(path + str(ep) + '/Control/' + str(i) + '.npy')

            # control
            for c in ctrls:
                data_ctrl.append(c)
            
            # point cloud
            count = 0
            for pts_perBatch in pts:
                # assert if there's large points
                assert(np.max(pts_perBatch) < np.sqrt(60.5**2 * 60.5**2))
                pts_filter = pts_filter[0:1900,:]
                data_pts.append(pts_filter)
                count += 1

    ## control: [throttle, steer, brake, speed]
    ctrl = np.asarray(data_ctrl, dtype=np.float32)
    ctrl = ctrl.reshape(ctrl.shape[0],  -1)
    # print('ctrl shape: ', ctrl.shape)                               # (N, 4)
    labels = ctrl[:, 1]  # steering                              # (N, 1)
    
    ## 3d point cloud
    points = np.asarray(data_pts, dtype=np.float32)                 # (N, 1900, 6)
    # print('points shape: ', points.shape) 
    
    ## check if labels are legit
    # solve the problem of having nan stored in steering right after respawn to new position
    mask2 = np.logical_not(np.isnan(labels))
    labels = labels[mask2]
    points = points[mask2]
    # print('ctrl shape after mask nan: ', labels.shape) 
    # print('points shape after mask nan: ', points.shape) 
    
    data_labels = torch.from_numpy(labels)  # steering
    data_points = torch.from_numpy(points)  # Lidar detection 3d points
    
    return data_points, data_labels


    
def generate_colour_lidar_data (path, episode, batch_count, batch_size):
    # episodes
    for ep in {episode}:
        for i in range(batch_count):
            all_batch_data = []
            ## test if the file exist
            filePath = path + str(ep) + '/Lidar/' + str(i) + '.npy'
            print(filePath)
            if not os.path.exists(filePath):
                print(filePath,'doesnt exist')
                print('its ok, skip')
                continue
            
            print('loading.. '+ filePath)
            lidar = np.load(filePath)
            # print('Shapes: Lidar: ', lidar.shape)   # (1000, 5000, 1, 3)
            # print('Data types: Lidar: ', lidar.dtype)
            lidar = lidar.reshape((lidar.shape[0], lidar.shape[1], 3))
            images = []
            for img_idx in range(NUM_IMG):
                imgpath = path + str(ep) + '/CameraRGB_' + str(img_idx) + '/' + str(i) + '.npy'
                images.append(np.load(imgpath))
            images = np.array(images)

            for index in range(batch_size):
                ## for Lidar data
                points = lidar[index]
                imgs = [images[0,index], images[1,index], images[2,index], images[3,index]]
                all_batch_data.append(colour_lidar(points, imgs))
                # print("all_batch_data length is: ", len(all_batch_data))

            all_batch_data = np.asarray(all_batch_data)
            # print("all_batch_data shape is: ", all_batch_data.shape)
            save_name = path + str(ep) + '/Lidar_Coloured/' + str(i) + '.npy'
            # print("combined lidar rgb saved in file: ", save_name)
            np.save(save_name, all_batch_data)
    print("generate coloured lidar point cloud is done")