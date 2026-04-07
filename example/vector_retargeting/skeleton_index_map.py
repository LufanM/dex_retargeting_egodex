# egodex index


EGODEX_INDEXS_DIC = {
    'rightHand': 0,
    'rightThumbKnuckle': 1, 'rightThumbIntermediateBase': 2, 'rightThumbIntermediateTip': 3, 'rightThumbTip': 4,
    'rightIndexFingerMetacarpal': 5, 'rightIndexFingerKnuckle': 6, 'rightIndexFingerIntermediateBase': 7, 'rightIndexFingerIntermediateTip': 8, 'rightIndexFingerTip': 9,
    'rightMiddleFingerMetacarpal': 10, 'rightMiddleFingerKnuckle': 11, 'rightMiddleFingerIntermediateBase': 12, 'rightMiddleFingerIntermediateTip': 13, 'rightMiddleFingerTip': 14, 
    'rightRingFingerMetacarpal': 15, 'rightRingFingerKnuckle': 16, 'rightRingFingerIntermediateBase': 17,'rightRingFingerIntermediateTip': 18, 'rightRingFingerTip': 19,
    'rightLittleFingerMetacarpal': 20, 'rightLittleFingerKnuckle': 21,'rightLittleFingerIntermediateBase': 22, 'rightLittleFingerIntermediateTip': 23,'rightLittleFingerTip': 24, 
    'rightForearm': 25,

    'leftHand': 26,
    'leftThumbKnuckle': 27, 'leftThumbIntermediateBase': 28,'leftThumbIntermediateTip': 29, 'leftThumbTip': 30,
    'leftIndexFingerMetacarpal': 31,'leftIndexFingerKnuckle': 32, 'leftIndexFingerIntermediateBase': 33,'leftIndexFingerIntermediateTip': 34, 'leftIndexFingerTip': 35,
    'leftMiddleFingerMetacarpal': 36, 'leftMiddleFingerKnuckle': 37,'leftMiddleFingerIntermediateBase': 38, 'leftMiddleFingerIntermediateTip': 39,'leftMiddleFingerTip': 40,
    'leftRingFingerMetacarpal': 41,'leftRingFingerKnuckle': 42, 'leftRingFingerIntermediateBase': 43, 'leftRingFingerIntermediateTip': 44, 'leftRingFingerTip': 45,
    'leftLittleFingerMetacarpal': 46, 'leftLittleFingerKnuckle': 47,'leftLittleFingerIntermediateBase': 48, 'leftLittleFingerIntermediateTip': 49,'leftLittleFingerTip': 50, 
    'leftForearm': 51
}

# mediapipe index 参考 该参数的 HandLandmark 类
RIGHT_INDEX_MAP_EGODEX2MANO = {
    # [0]: EGODEX的索引，[1]: mediapipe 的索引
    [0,  1, 2, 3, 4,    6, 7, 8, 9,    11, 12, 13, 14,   16, 17, 18, 19,    21, 22, 23, 24],
    [0,  1, 2, 3, 4,    5, 6, 7, 8,    9, 10, 11, 12,    13, 14, 15, 16,    17, 18, 19, 20]
}

LEFT_INDEX_MAP_EGODEX2MANO = {
    # [0]: EGODEX的索引，[1]: mediapipe 的索引
    [26,    27, 28, 29, 30,    32, 33, 34, 35,    37, 38, 39, 40,    42, 43, 44, 45,    47, 48, 49, 50],
    [0,     1,  2,  3,  4,     5,  6,  7,  8,     9,  10, 11, 12,    13, 14, 15, 16,    17, 18, 19, 20]
}              