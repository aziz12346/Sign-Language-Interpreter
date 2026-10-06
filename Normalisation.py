import pandas as pd

input_file = r"C:\Development\leapc-python-bindings-main\Dataset\Handsigns.csv"
output_file = r"C:\Development\leapc-python-bindings-main\Dataset\Handsigns_preprocessed1.csv"

means_file = r"C:\Development\leapc-python-bindings-main\Dataset\zscore_means.csv"
stds_file = r"C:\Development\leapc-python-bindings-main\Dataset\zscore_stds.csv"

df = pd.read_csv(input_file, header=None)

# Metadata
label = 0
frame = 1
time_us = 2
time_s = 3
num_hands = 4

each_hand = 148 # No. columns for each hand
hand1_start = 5
hand2_start = hand1_start + each_hand   # 153

# Hand block end columns
hand1_end = hand1_start + each_hand - 1   # 152
hand2_end = hand2_start + each_hand - 1   # 300

# Palm columns
palm1_x = hand1_start + 2   # 7
palm1_y = hand1_start + 3   # 8
palm1_z = hand1_start + 4   # 9

palm2_x = hand2_start + 2   # 155
palm2_y = hand2_start + 3   # 156
palm2_z = hand2_start + 4   # 157


df.loc[df[num_hands] == 1, hand2_start:hand2_end] = 0


def feature_cols(hand_start):
    cols = []

    finger_start = hand_start + 8

    each_finger = 28

    # Palm normal columns
    normal_cols = [hand_start + 5, hand_start + 6, hand_start + 7]

    for finger_id in range(5):
        start = finger_start + finger_id * each_finger # Thumb = 13   Index = 41     Middle = 69     Ring = 97     Pinky = 125

        # Tip posisitons (x,y,z)
        cols.extend([start, start + 1, start + 2]) # 13, 14, 15

        # Bone joints
        cols.extend(range(start + 3, start + 27)) # 16-40

    return cols, normal_cols

hand1_cols, hand1_normal_cols = feature_cols(hand1_start)
hand2_cols, hand2_normal_cols = feature_cols(hand2_start)

# Palm-relative Normalisation
# Hand 1
for i in range(0, len(hand1_cols), 3):
    x_col = hand1_cols[i]
    y_col = hand1_cols[i + 1]
    z_col = hand1_cols[i + 2]

    df[x_col] = df[x_col] - df[palm1_x]
    df[y_col] = df[y_col] - df[palm1_y]
    df[z_col] = df[z_col] - df[palm1_z]

df[palm1_x] = 0
df[palm1_y] = 0
df[palm1_z] = 0


# Hand 2
for i in range(0, len(hand2_cols), 3):
    x_col = hand2_cols[i]
    y_col = hand2_cols[i + 1]
    z_col = hand2_cols[i + 2]

    df[x_col] = df[x_col] - df[palm2_x]
    df[y_col] = df[y_col] - df[palm2_y]
    df[z_col] = df[z_col] - df[palm2_z]

if palm2_z < df.shape[1]:
    df[palm2_x] = 0
    df[palm2_y] = 0
    df[palm2_z] = 0


# Z-score Normalisation
zscore_cols = (
    hand1_cols + hand1_normal_cols + hand2_cols + hand2_normal_cols
)

zscore_cols = [col for col in zscore_cols if col < df.shape[1]]

means = df[zscore_cols].mean()
stds = df[zscore_cols].std()

stds = stds.replace(0, 1)

means.to_csv(means_file, index=False, header=False)
stds.to_csv(stds_file, index=False, header=False)

print("Saved z-score means to:", means_file)
print("Saved z-score stds to:", stds_file)

df[zscore_cols] = (df[zscore_cols] - means) / stds



df.to_csv(output_file, index=False, header=False)
