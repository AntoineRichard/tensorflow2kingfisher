# ROS DREAMER Agents

This ROS package features 3 types of Reinforcement Learning agents:
 - Projection to Projection agents (referred to as P2P in our paper), here they
   are named *cmap2cmap*.
 - Laser to Projection agents (referred to as L2P in our paper), here, they are
   named *lzr2cmap*.
 - Laser to Laser agents (referred to as L2L in our paper), here, they are
   named *lzr2lzr*.

For each of the type 2 nodes can be ran:
 - dreamer : this is the code used to run at training time. It records
   episodes, plays for a limited amount of steps, applies exploration or not
   and supports random agent.
 - agent : this is the code used to run at evaluation time. It plays
   indefinitely.

TODO: add launch files
