import numpy as np

def create_get_features(types):
    def onebyone_onehot(observation, branch, action):
        """
        One hot encoding of 1x1 tiles and action taken
        """
        feature_vector_player = np.zeros(600)
        z = observation[2] + 7.5
        x = observation[0] + 15
        feature_vector_player[int(x*15 + z)] = 1

        feature_vector_ball = np.zeros(600)
        z = observation[5] + 7.5
        x = observation[3] + 15
        feature_vector_ball[int(x*15 + z)] = 1

        feature_vector = np.append(feature_vector_player, feature_vector_ball)

        action_1 = np.array([0, 0, 0])
        action_2 = np.array([0, 0, 0])
        if branch == 0:
            action_1[action] = 1
        elif branch == 1:
            action_2[action] = 1
        feature_vector = np.append(feature_vector, action_1)
        feature_vector = np.append(feature_vector, action_2)

        return feature_vector

    def onebyone_onehot_includeMore(observation, branch, action):
        """
        One hot encoding of 1x1 tiles and action taken
        """
        feature_vector_player = np.zeros(600)
        z = observation[2] + 7.5
        x = observation[0] + 15
        feature_vector_player[int(x*15 + z)] = 1

        feature_vector_ball = np.zeros(600)
        z = observation[5] + 7.5
        x = observation[3] + 15
        feature_vector_ball[int(x*15 + z)] = 1

        feature_vector = np.append(feature_vector_player, feature_vector_ball)

        action_1 = np.array([0, 0, 0])
        action_2 = np.array([0, 0, 0])
        if branch == 0:
            action_1[action] = 1
        elif branch == 1:
            action_2[action] = 1
        feature_vector = np.append(feature_vector, action_1)
        feature_vector = np.append(feature_vector, action_2)

        more_vector = np.array([observation[6], observation[7], observation[8], observation[9], observation[10], observation[11]])

        feature_vector = np.append(feature_vector, more_vector)

        return feature_vector

    def onebyone_onehot_includeMore_overlapping(observation, branch, action):
        """
        One hot encoding of 1x1 tiles and action taken
        """
        feature_vector = np.zeros(600)
        z = observation[2] + 7.5
        x = observation[0] + 15
        feature_vector[int(x*15 + z)] += 1

        z = observation[5] + 7.5
        x = observation[3] + 15
        feature_vector[int(x*15 + z)] += 1

        action_1 = np.array([0, 0, 0])
        action_2 = np.array([0, 0, 0])
        if branch == 0:
            action_1[action] = 1
        elif branch == 1:
            action_2[action] = 1
        feature_vector = np.append(feature_vector, action_1)
        feature_vector = np.append(feature_vector, action_2)

        more_vector = np.array([observation[6], observation[7], observation[8], observation[9], observation[10], observation[11]])

        feature_vector = np.append(feature_vector, more_vector)

        return feature_vector

    def everypossiblecombo(observation, branch, action):
        feature_vector = np.zeros(360000)
        player_z = observation[2] + 7.5
        player_x = observation[0] + 15
        ball_z = observation[5] + 7.5
        ball_x = observation[3] + 15

        feature_vector[(int(player_x*15 + player_z) * 600 + int(ball_x*15 + ball_z))] += 1

        action_1 = np.array([0, 0, 0])
        action_2 = np.array([0, 0, 0])
        if branch == 0:
            action_1[action] = 1
        elif branch == 1:
            action_2[action] = 1
        feature_vector = np.append(feature_vector, action_1)
        feature_vector = np.append(feature_vector, action_2)

        more_vector = np.array([observation[6], observation[7], observation[8], observation[9], observation[10], observation[11]])

        feature_vector = np.append(feature_vector, more_vector)

        return feature_vector

    def original(observation, branch, action):
        """
        original observation + onehot of actions
        """
        action_1 = np.array([0, 0, 0])
        action_2 = np.array([0, 0, 0])
        if branch == 0:
            action_1[action] = 1
        elif branch == 1:
            action_2[action] = 1
        feature_vector = np.append(observation, action_1)
        feature_vector = np.append(feature_vector, action_2)

        return feature_vector

    if types == "onebyone_onehot":
        return onebyone_onehot
    elif types == "onebyone_onehot_includeMore":
        return onebyone_onehot_includeMore
    elif types == "onebyone_onehot_includeMore_special":
        return onebyone_onehot_includeMore_overlapping
    elif types == 'original':
        return original
    elif types == 'best':
        return everypossiblecombo