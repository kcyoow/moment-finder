# @title questions: the criteria under test (shared by image and video experiments)
"""Question sets. Each frame/image is one request; several criteria share that request."""

FINGER_HEART_TEXT = "Is anyone in `frame` making a finger heart gesture?"
FINGER_HEART_TRUE = ("At least one hand shows the Korean finger heart: the thumb and index finger are crossed "
                     "so their tips form a small heart. One hand is enough; both hands doing it also counts.")
FINGER_HEART_FALSE = ("No hand shows that gesture. Other gestures do not count, including a big heart made with "
                      "both hands together, waving, a V sign, a thumbs up, an OK sign, a fist, or hands that are "
                      "not visible.")
TWO_HAND_HEART_TEXT = "Is anyone in `frame` forming one big heart shape with both hands together?"


def video_questions():
    return {
        "fh_plain": Noul(instructions=FINGER_HEART_TEXT),
        "fh_crit": Noul(instructions=FINGER_HEART_TEXT,
                        criteria=NoulCriteria(true=FINGER_HEART_TRUE, false=FINGER_HEART_FALSE)),
        "two_hand_heart": Noul(instructions=TWO_HAND_HEART_TEXT),
    }


print("questions ready")
