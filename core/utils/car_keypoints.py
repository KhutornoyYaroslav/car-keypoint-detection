car_nodes = [
    "fl wheel", # 0
    "bl wheel", # 1
    "fr wheel", # 2
    "br wheel", # 3
    "windshield tr", # 4
    "windshield tl", # 5
    "windshield bl", # 6
    "windshield br", # 7
    "rear window tl", # 8
    "rear window tr", # 9
    "rear window br", # 10
    "rear window bl", # 11
    "rearview mirror r", # 12
    "rearview mirror l", # 13
    "bottom of license fr", # 14
    "bottom of license fl", # 15
    "bottom of license bl", # 16
    "bottom of license br", # 17
    "headlight fr inner bottom", # 18
    "headlight fr outer top", # 19
    "headlight fl inner bottom", # 20
    "headlight fl outer top", # 21
    "headlight bl inner bottom", # 22
    "headlight bl outer top", # 23
    "headlight br inner bottom", # 24
    "headlight br outer top", # 25
    "bottom bumper fr", # 26
    "bottom bumper fl", # 27
    "bottom bumper bl", # 28
    "bottom bumper br", # 29
    "side window back r", # 30
    "side window back l", # 31
]
car_edges = [
    # Face
    [14, 26], [14, 18],
    [15, 27], [15, 20],
    [14, 15],
    [26, 18], [26, 19], [18, 19],
    [27, 20], [27, 21], [20, 21],
    [18, 20],
    [19, 21],
    # Windshield and rearview mirrors
    [4, 5], [5, 6], [6, 7], [7, 4],
    [12, 4], [12, 7],
    [13, 5], [13, 6],
    # Face <-> Windshield
    [19, 7], [21, 6],
    # Back
    [16, 28], [16, 22],
    [17, 29], [17, 24],
    [16, 17],
    [28, 22], [28, 23], [22, 23],
    [29, 24], [29, 25], [24, 25],
    [22, 24],
    [23, 25],
    # Rear window
    [8, 9], [9, 10], [10, 11], [11, 8],
    # Back <-> Rear window
    [23, 11], [25, 10],
    # Face <-> Back (roof part)
    [4, 9], [5, 8],
    # Wheels
    [0, 1], [2, 3],
    # Front wheels <-> Face and rearview mirrors
    [0, 27], [0, 21], [0, 13],
    [2, 26], [2, 19], [2, 12],
    # Back wheels <-> Back and rearview mirrors
    [1, 13], [1, 31], [1, 28],
    [3, 12], [3, 30], [3, 29],
    # Side window <-> Rear window
    [30, 10], [31, 11],
    # Side window <-> Rearview mirrors
    [30, 12], [31, 13]
]
