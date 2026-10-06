import leap
import time
import numpy as np
import pandas as pd
from tensorflow import keras


class MyListener(leap.Listener):
    def __init__(self):
        print("Demo initialized")

        super().__init__()
        self.last_print_time = 0.0

        # Load CNN model
        self.model_path = r"C:\Development\leapc-python-bindings-main\CNN model\BSL_CNN.keras"
        self.model = keras.models.load_model(self.model_path)

        # Load z-score means and standard deviations
        self.means_file = r"C:\Development\leapc-python-bindings-main\Dataset\zscore_means.csv"
        self.stds_file = r"C:\Development\leapc-python-bindings-main\Dataset\zscore_stds.csv"

        self.means = pd.read_csv(self.means_file, header=None).iloc[:, 0].to_numpy(dtype="float32")
        self.stds = pd.read_csv(self.stds_file, header=None).iloc[:, 0].to_numpy(dtype="float32")

        # Label mapping
        self.label_map = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "I",
            8: "K",
            9: "L"
        }

        # Column layout
        self.label = 0
        self.frame = 1
        self.time_us = 2
        self.time_s = 3
        self.num_hands = 4

        self.each_hand = 148
        self.hand1_start = 5
        self.hand2_start = self.hand1_start + self.each_hand

        self.hand1_end = self.hand1_start + self.each_hand - 1
        self.hand2_end = self.hand2_start + self.each_hand - 1

        self.palm1_x = self.hand1_start + 2
        self.palm1_y = self.hand1_start + 3
        self.palm1_z = self.hand1_start + 4

        self.palm2_x = self.hand2_start + 2
        self.palm2_y = self.hand2_start + 3
        self.palm2_z = self.hand2_start + 4

        #Z-score columns
        self.hand1_cols, self.hand1_normal_cols = self.feature_cols(self.hand1_start)
        self.hand2_cols, self.hand2_normal_cols = self.feature_cols(self.hand2_start)

        self.zscore_cols = (
            self.hand1_cols + self.hand1_normal_cols +
            self.hand2_cols + self.hand2_normal_cols
        )

        print("CNN model loaded")

    def on_connection_event(self, event):
        print("Leap Service Connected")


    def on_connection_lost_event(self, event):
        print("Leap Service Disconnected")


    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print("Connected")
        print(f"Found device {info.serial}")


    def on_device_lost_event(self, event):
        print("Disconnected")


    def on_exit_event(self, event):
        print("Exited")


    # Normalisation.py
    def feature_cols(self, hand_start):
        cols = []

        finger_start = hand_start + 8
        each_finger = 28

        # Palm normal columns
        normal_cols = [hand_start + 5, hand_start + 6, hand_start + 7]

        for finger_id in range(5):
            start = finger_start + finger_id * each_finger

            # Tip positions (x,y,z)
            cols.extend([start, start + 1, start + 2])

            # Bone joints
            cols.extend(range(start + 3, start + 27))

        return cols, normal_cols


    def on_tracking_event(self, event):
        current_time = time.time()

        if current_time - self.last_print_time < 0.5:
            return

        self.last_print_time = current_time

        # Must detect at least one hand
        if len(event.hands) < 1:
            return

        # Timestamp
        ts_us = event.info.timestamp
        ts_s = ts_us / 1_000_000.0

        # Sample
        output = []

        # Dummy label placeholder
        output.append("-1")

        # Frame info
        output.append(str(event.tracking_frame_id))
        output.append(str(ts_us))
        output.append(str(ts_s))
        output.append(str(len(event.hands)))

        # Hand info (type, confidence, palm, fingers and bones)
        for hand in event.hands:
            hand_type = "0" if str(hand.type) == "HandType.Left" else "1"
            hand_confidence = hand.confidence
            normal = hand.palm.normal
            palm = hand.palm.position

            output.append(hand_type)
            output.append(str(hand_confidence))

            output.extend([
                str(palm.x), str(palm.y), str(palm.z),
                str(normal.x), str(normal.y), str(normal.z)
            ])

            # Fingers and Bones
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']

            for i, digit in enumerate(hand.digits):
                name = finger_names[i] if i < len(finger_names) else f"digit_{i}"

                # Fingertip positions
                tip = digit.distal.next_joint
                output.extend([str(tip.x), str(tip.y), str(tip.z)])

                bones = [digit.metacarpal, digit.proximal, digit.intermediate, digit.distal]

                for bone_name, bone in zip(bone_names, bones):
                    pj = bone.prev_joint
                    nj = bone.next_joint

                    output.extend([
                        str(pj.x), str(pj.y), str(pj.z),
                        str(nj.x), str(nj.y), str(nj.z)
                    ])

                # Extended state
                is_extended = digit.is_extended
                output.append("1" if is_extended else "0")

        # No second hand
        while len(output) < 301:
            output.append("0")

        row = ",".join(output)

        # Convert row to single-row DataFrame
        df = pd.DataFrame([row.split(",")], dtype="float32")

        
        # Normalisation.py again
        # Zero second hand if only one hand
        df.loc[df[self.num_hands] == 1, self.hand2_start:self.hand2_end] = 0

        # Palm-relative Normalisation
        # Hand 1
        for i in range(0, len(self.hand1_cols), 3):
            x_col = self.hand1_cols[i]
            y_col = self.hand1_cols[i + 1]
            z_col = self.hand1_cols[i + 2]

            df[x_col] = df[x_col] - df[self.palm1_x]
            df[y_col] = df[y_col] - df[self.palm1_y]
            df[z_col] = df[z_col] - df[self.palm1_z]

        df[self.palm1_x] = 0
        df[self.palm1_y] = 0
        df[self.palm1_z] = 0

        # Hand 2
        for i in range(0, len(self.hand2_cols), 3):
            x_col = self.hand2_cols[i]
            y_col = self.hand2_cols[i + 1]
            z_col = self.hand2_cols[i + 2]

            if z_col >= df.shape[1]:
                break

            df[x_col] = df[x_col] - df[self.palm2_x]
            df[y_col] = df[y_col] - df[self.palm2_y]
            df[z_col] = df[z_col] - df[self.palm2_z]

        if self.palm2_z < df.shape[1]:
            df[self.palm2_x] = 0
            df[self.palm2_y] = 0
            df[self.palm2_z] = 0

        
        features = df.iloc[:, 4:].copy()

        # Z-score Normalisation
        zscore_feature_cols = [col - 4 for col in self.zscore_cols]

        feature_array = features.to_numpy(dtype="float32")

        feature_array[0, zscore_feature_cols] = (
            feature_array[0, zscore_feature_cols] - self.means
        ) / self.stds

        # Reshape 
        feature_array = feature_array.reshape(1, feature_array.shape[1], 1)

        # Prediction
        predictions = self.model.predict(feature_array, verbose=0)

        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        predicted_label = self.label_map[predicted_class]

        print("--------------------------------")
        print("Predicted Sign:", predicted_label)
        print("Confidence:", round(confidence, 4))
        print("--------------------------------")


def main():
    my_listener = MyListener()

    connection = leap.Connection()
    connection.add_listener(my_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        while running:
            time.sleep(1)


if __name__ == "__main__":
    main()

