## NOTE: THis code gets the position of Allegro hand and Kinova arm
from holobot.robot.allegro.allegro import AllegroHand
from holobot.robot.kinova import KinovaArm

hand = AllegroHand()
arm = KinovaArm()

#get the allegro joint states
print("Kinova arm cartesian position:{}".format(arm.get_cartesian_state()))
print("Allegro hand joint states:{}".format(hand.get_joint_state()))
