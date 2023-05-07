topics_name = {
    # action publish topic
    "car1_action_pub": '/CAR1/cmd_vel',
    "car2_action_pub": '/CAR2/cmd_vel',
    "car3_action_pub": '/CAR3/cmd_vel',
    "car4_action_pub": '/CAR4/cmd_vel',
    # pos publish topic (to rviz)
    "car1_pos_pub": '/CAR1/scaled_pos',
    "car2_pos_pub": '/CAR2/scaled_pos',
    "car3_pos_pub": '/CAR3/scaled_pos',
    "car4_pos_pub": '/CAR4/scaled_pos',
    "obj_pos_pub":  '/obj/scaled_pos',
    "obs_pos_pub": '/obs/scaled_pos',
    "tar_pos_pub": '/target/scaled_pos',
    # pos subscribe topic (from motion capture)
    "car1_pos_sub": '/vrpn_client_node/CAR1Final/pose',
    "car2_pos_sub": '/vrpn_client_node/CAR2Final/pose',
    "car3_pos_sub": '/vrpn_client_node/CAR3Final/pose',
    "car4_pos_sub": '/vrpn_client_node/CAR4Final/pose',
    "obj_pos_sub": '/vrpn_client_node/OBJ/pose',
    "obs_pos_sub": '/vrpn_client_node/obstacle/pose'
}