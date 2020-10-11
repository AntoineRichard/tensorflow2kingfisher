#!/usr/bin/env python

import rospy

import gym
import uuid
import io
import pathlib
import datetime
import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from std_msgs.msg import Float32
from std_msgs.msg import Bool
from rl_server.msg import Episode
from heron_msgs.msg import Drive
from sensor_msgs.msg import Image

import much_simpler_dreamer
import tools

class DreamerAgent:
    def __init__(self):
        self.agent = None
        self.save_directory = '/mnt/nvme-storage/antoine/DREAMER/dreamer/logdir/kf_sim/dreamer/21_float32_trauma/'
        self.Done = True
	self.precision = 32
        self.max_steps = 1000
        self.obs = {}
        # ROS
        self.drive = Drive()
        self.action_pub_ = rospy.Publisher('/cmd_rl', Drive, queue_size=1)
        self.initialize_agent()
        self.refresh_agent()
        rospy.Subscriber("/reward_generator/DreamersView", Image, self.ImageCallback, queue_size=1)
        rospy.Subscriber("/reward_generator/reward", Float32, self.rewardCallback, queue_size=1)

    def initialize_agent(self):
        parser = argparse.ArgumentParser()
        for key, value in much_simpler_dreamer.define_config().items():
            parser.add_argument('--'+str(key), type=tools.args_type(value), default=value)
        config, unknown = parser.parse_known_args()
        if config.gpu_growth:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        if config.precision == 16:
            prec.set_policy(prec.Policy('mixed_float16'))
        config.steps = int(config.steps)

        actspace = gym.spaces.Box(np.array([-1,-1]),np.array([1,1]))
        self.agent = much_simpler_dreamer.Dreamer(config, actspace)
        if pathlib.Path(self.save_directory+'/variables.pkl').exists():
            print('Load checkpoint.')
            self.agent.load(self.save_directory)
        else:
            raise ValueError('Could not load weights')
        self.state = None

    def refresh_agent(self):
        self.state = None
        if pathlib.Path(self.save_directory+'/variables.pkl').exists():
            print('Load checkpoint.')
            self.agent.load(self.save_directory)
        else:
            raise ValueError('Could not load weights')
        self.obs['image'] = np.zeros((1,64,64,3),dtype=np.uint8)
        self.obs['reward'] = np.zeros((1))
        self.Done = False

    def ImageCallback(self, obs):
        self.image = np.reshape(np.fromstring(obs.data, np.uint8),[64,64,3])
        if not self.Done:
            self.obs['image'][0] = self.image
            self.obs['reward'][0] = self.reward
            t_actions, self.state = self.agent.policy(self.obs, self.state, False)
            embed = self.agent.policy2(self.obs, self.state, False)
            actions = np.array(t_actions)[0]
            self.action_pub_.publish(self.actions2Twist(actions))
            self.actions = actions
    
    def rewardCallback(self, reward):
        if not self.Done:
            self.reward = reward.data


    def actions2Twist(self, actions):
        self.drive.left = actions[0]
        self.drive.right = actions[1]
        return self.drive

if __name__ == "__main__":
    rospy.init_node('dreamer_agent')
    DA = DreamerAgent()
    rospy.spin()

