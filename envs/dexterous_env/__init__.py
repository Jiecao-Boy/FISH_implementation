from gym.envs.registration import register 


register(
	id='CubeFlipping-v1',
	entry_point='dexterous_env.cube_flipping_env:CubeFlippingEnv',
	max_episode_steps=76,
)