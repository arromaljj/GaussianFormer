


"""
Camera Images: 
    CAM_FRONT
    CAM_FRONT_RIGHT
    CAM_FRONT_LEFT
    CAM_BACK
    CAM_BACK_LEFT
    CAM_BACK_RIGHT

Image: 
    Resolution: 1600 x 900
    mean: [123.675, 116.28, 103.53] 
    std: [58.395, 57.12, 57.375]
    format: rgb

Camera Intrinsics: 
    3x3 camera intrinsic matrix
    Width and Height of each camera image

Camera Extrinsics: 
    Camera-to-ego transforms (Basically tf data)
        Camera position relative to vehicle ego frame
        Rotation Quaternion: Camera orientation

Vehicle Ego Pose: 
    Translation vector: Vehicle Position in global frame
    Rotation Quaternion: Vehicle Orientation in global frame

    

"""