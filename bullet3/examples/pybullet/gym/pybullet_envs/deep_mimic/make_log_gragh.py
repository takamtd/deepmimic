import pandas as pd
import matplotlib.pyplot as plt
import os

err_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data//err/err_log0.csv"
sim_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/sim/sim_log0.csv"
kin_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/kin/kin_log0.csv"
sim_angle_log_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/sim_angle/sim_angle_log0.csv"
gragh_path = "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/gragh/angle/"
os.makedirs(gragh_path)
err_data = pd.read_csv(err_log_path)
sim_data = pd.read_csv(sim_log_path)
kin_data = pd.read_csv(kin_log_path)
sim_angle_data = pd.read_csv(sim_angle_log_path)
# for index, column_name in enumerate(err_data):
#     if index != 0:
#         # print(column_name)
#         fig = plt.figure(index)
#         plt.plot(err_data[column_name], linestyle = "-")
#         plt.title(column_name)
#         plt.xlabel("timestep")
#         plt.ylabel("loss")
#         curr_gragh_path = gragh_path + "{}.png".format(column_name)
#         plt.savefig(curr_gragh_path)

for index, column_name in enumerate(sim_data):
    if index != 0:
        # print(column_name)
        fig = plt.figure(index)
        plt.plot(sim_data[column_name], linestyle = "-", label = "atlas")
        plt.plot(kin_data[column_name], linestyle = "--", label = "mocap")
        # plt.plot(kin_data[column_name]*1.5, linestyle = "--", label = "mocap*1.5")
        plt.legend(loc = "upper right")
        plt.title(column_name)
        plt.xlabel("timestep")
        plt.ylabel("angle")
        curr_gragh_path = gragh_path + "{}.png".format(column_name)
        plt.savefig(curr_gragh_path)

# # column_name = "r_leg_hpy_orn_r"
# # column_name = "r_leg_hpy_orn_y"
# # column_name = "root_orn_r"
# # column_name = "root_orn_y"
# column_name = "l_arm_elx_angle"
# # column_name2 = "r_arm_elx_angle"
# plt.plot(sim_data[column_name], linestyle = "-")
# plt.plot(kin_data[column_name], linestyle = "--")
# # print(sim_angle_data[column_name])
# # plt.plot(sim_angle_data[column_name2], linestyle = "--")
# plt.title(column_name)
# plt.xlabel("timestep")
# plt.ylabel("angle")
# plt.show()
# curr_gragh_path = gragh_path + "{}.png".format(column_name)
# plt.savefig(curr_gragh_path)

"""
    [
        "root_pos_x", "root_pos_y", "root_pos_z", "root_vel_x", "root_vel_y", "root_vel_z",
        "root_orn_r", "root_orn_p", "root_orn_y", "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
        "com_pos_x", "com_pos_y", "com_pos_z", "com_vel_x", "com_vel_y", "com_vel_z",
        "back_bkz_orn_r", "back_bkz_orn_p", "back_bkz_orn_y", "back_bkz_vel_x", "back_bkz_vel_y", "back_bkz_vel_z",
        "back_bkx_orn_r", "back_bkx_orn_p", "back_bkx_orn_y", "back_bkx_vel_x", "back_bkx_vel_y", "back_bkx_vel_z",
        "neck_ry_orn_r", "neck_ry_orn_p", "neck_ry_orn_y", "neck_ry_vel_x", "neck_ry_vel_y", "neck_ry_vel_z",
        "r_leg_hpy_orn_r", "r_leg_hpy_orn_p", "r_leg_hpy_orn_y", "r_leg_hpy_vel_x", "r_leg_hpy_vel_y", "r_leg_hpy_vel_z",
        "r_leg_kny_angle", "r_leg_kny_vel",
        "r_leg_akx_orn_r", "r_leg_akx_orn_p", "r_leg_akx_orn_y", "r_leg_akx_vel_x", "r_leg_akx_vel_y", "r_leg_akx_vel_z",
        "r_arm_ely_orn_r", "r_arm_ely_orn_p", "r_arm_ely_orn_y", "r_arm_ely_vel_x", "r_arm_ely_vel_y", "r_arm_ely_vel_z",
        "r_arm_elx_angle", "r_arm_elx_vel",
        "l_leg_hpy_orn_r", "l_leg_hpy_orn_p", "l_leg_hpy_orn_y", "l_leg_hpy_vel_x", "l_leg_hpy_vel_y", "l_leg_hpy_vel_z",
        "l_leg_kny_angle", "l_leg_kny_vel",
        "l_leg_akx_orn_r", "l_leg_akx_orn_p", "l_leg_akx_orn_y", "l_leg_akx_vel_x", "l_leg_akx_vel_y", "l_leg_akx_vel_z",
        "l_arm_ely_orn_r", "l_arm_ely_orn_p", "l_arm_ely_orn_y", "l_arm_ely_vel_x", "l_arm_ely_vel_y", "l_arm_ely_vel_z",
        "l_arm_elx_angle", "l_arm_elx_vel",
        "l_leg_akx_pos_x", "l_leg_akx_pos_y", "l_leg_akx_pos_z",
        "r_leg_akx_pos_x", "r_leg_akx_pos_y", "r_leg_akx_pos_z"
    ]
    [
        "back_bkz_angle", "back_bkz_vel", "back_bky_angle", "back_bky_vel", "back_bkx_angle", "back_bkx_vel",
        "l_arm_shz_angle", "l_arm_shz_vel", "l_arm_shx_angle", "l_arm_shx_vel",
        "l_arm_ely_angle", "l_arm_ely_vel", "l_arm_elx_angle", "l_arm_elx_vel",
        "l_arm_wry_angle", "l_arm_wry_vel", "l_arm_wrx_angle", "l_arm_wrx_vel", "l_arm_wry2_angle", "l_arm_wry2_vel",
        "neck_ry_angle", "neck_ry_vel",
        "r_arm_shz_angle", "r_arm_shz_vel", "r_arm_shx_angle", "r_arm_shx_vel",
        "r_arm_ely_angle", "r_arm_ely_vel", "r_arm_elx_angle", "r_arm_elx_vel",
        "r_arm_wry_angle", "r_arm_wry_vel", "r_arm_wrx_angle", "r_arm_wrx_vel", "r_arm_wry2_angle", "r_arm_wry2_vel",
        "l_leg_hpz_angle", "l_leg_hpz_vel", "l_leg_hpx_angle", "l_leg_hpx_vel", "l_leg_hpy_angle", "l_leg_hpy_vel",
        "l_leg_kny_angle", "l_leg_kny_vel",
        "l_leg_aky_angle", "l_leg_aky_vel", "l_leg_akx_angle", "l_leg_akx_vel",
        "r_leg_hpz_angle", "r_leg_hpz_vel", "r_leg_hpx_angle", "r_leg_hpx_vel", "r_leg_hpy_angle", "r_leg_hpy_vel",
        "r_leg_kny_angle", "r_leg_kny_vel",
        "r_leg_aky_angle", "r_leg_aky_vel", "r_leg_akx_angle", "r_leg_akx_vel"
    ]
"""