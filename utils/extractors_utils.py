import pandas as pd
import cv2
import tensorflow as tf

def extract_kpts_to_df_movenet(df, image_path,  movenet_model):
    image = cv2.imread(image_path)
    sh0 = image.shape[0]
    sh1 = image.shape[0]
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    
    outputs = movenet_model(image)
# Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    input_kpts = None

    if keypoints is not None:
        
        first_list = [image_path]

        for elem in keypoints[0,0]:
            elem_np = np.array(elem)
            
            elem_np[0] = elem_np[0] * sh0
            elem_np[1] = elem_np[1] * sh1
            
            first_list.append(elem_np.tolist())

        kpts_df = pd.DataFrame([first_list], columns = ["im_path", "nose","leftEye", "rightEye", "leftEar",
                                              "rightEar","leftShoulder","rightShoulder","leftElbow",
                                              "rightElbow", "leftWrist","rightWrist","leftHip", 
                                              "rightHip","leftKnee","rightKnee", "leftAnkle", "rightAnkle"])

        df = pd.concat([df, kpts_df])
    else:
        first_list = [image_path]
        for i in range(17):
            first_list.append("NaN")
        kpts_df = pd.DataFrame([first_list], columns = ["im_path", "nose","leftEye", "rightEye", "leftEar",
                                              "rightEar","leftShoulder","rightShoulder","leftElbow",
                                              "rightElbow", "leftWrist","rightWrist","leftHip", 
                                              "rightHip","leftKnee","rightKnee", "leftAnkle", "rightAnkle"])
        df = pd.concat([df, kpts_df])
    
    return df       
    
