def detect_abnormal(values):

    ranges = {
        "hemoglobin": (13,17),
        "wbc": (4000,11000),
        "rbc": (4.5,6),
        "platelets": (150000,400000),
        "mcv": (80,100),
        "mch": (27,33),
        "mchc": (31,36),
        "neutrophils": (40,75),
        "lymphocytes": (20,45),
        "monocytes": (2,10),
        "eosinophils": (0,6),
        "basophils": (0,2)
    }

    abnormal = {}

    for k,v in values.items():

        if k in ranges:

            low,high = ranges[k]

            if v < low:
                abnormal[k] = "Low"
            elif v > high:
                abnormal[k] = "High"

    return abnormal